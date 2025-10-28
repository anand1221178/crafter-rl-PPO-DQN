# icm_lite_wrapper.py
"""
ICM-lite curiosity wrapper for DQN exploration (Gen-4).
Train-only intrinsic reward with per-episode cap and progress decay.
OFF at evaluation to maintain clean Crafter metrics.
"""
import math
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities ----------
class RunningMeanStd:
    """Welford running mean/var for standardizing forward error."""
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float):
        """Update with a scalar (Python float)."""
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2  # unnormalized
        # keep var positive
        self.var = max(self.var, 1e-8)

    @property
    def std(self):
        return math.sqrt(self.var / max(self.count - 1.0, 1.0))


def to_chw_float(obs_np: np.ndarray) -> torch.Tensor:
    """Convert (H,W,C) uint8 numpy -> (1,C,H,W) float32 torch in [0,1]."""
    t = torch.from_numpy(obs_np).permute(2, 0, 1).contiguous().float() / 255.0
    return t.unsqueeze(0)


def one_hot(a: int, n: int, device):
    """Create one-hot action vector."""
    v = torch.zeros((1, n), device=device)
    v[0, a] = 1.0
    return v


# ---------- Tiny ICM networks (separate from DQN!) ----------
class IcmEncoder(nn.Module):
    """Small conv encoder -> 128-d embedding. Fast + stable."""
    def __init__(self, in_ch: int = 3, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(inplace=True),   # 64->32
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),      # 32->16
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),      # 16->8
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        return self.net(x)


class IcmForward(nn.Module):
    """Predict z_{t+1} from (z_t, a_one_hot)."""
    def __init__(self, emb_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + n_actions, 256), nn.ReLU(inplace=True),
            nn.Linear(256, emb_dim)
        )

    def forward(self, z_t, a_oh):
        x = torch.cat([z_t, a_oh], dim=1)
        return self.net(x)


class IcmInverse(nn.Module):
    """(Optional) Predict action from (z_t, z_{t+1}) to make features action-relevant."""
    def __init__(self, emb_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim*2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, z_t, z_tp1):
        x = torch.cat([z_t, z_tp1], dim=1)
        return self.net(x)  # logits


# ---------- The wrapper ----------
class IcmLiteWrapper(gym.Wrapper):
    """
    Train-time curiosity reward for exploration.
    * Computes intrinsic r_int online per step and ADDS it to env reward.
    * Per-episode CAP (e.g., 0.31) and PROGRESS DECAY (20% -> 60% of training).
    * OFF at evaluation (enabled=False) and can also skip updates (optimize=False).
    """
    def __init__(
        self,
        env,
        n_actions: int,
        total_steps: int = 1_000_000,
        beta: float = 0.05,              # base scale for intrinsic
        cap_per_ep: float = 0.31,        # ~10% of Gen-1 median return
        decay_start: float = 0.20,       # start decay at 20% of training
        decay_stop: float = 0.60,        # finish decay by 60%
        use_inverse: bool = True,
        lr: float = 1e-3,
        max_grad_norm: float = 5.0,
        emb_dim: int = 128,
        ebar_clip: float = 3.0,          # clip standardized error before scaling
        device: str | None = None,
        enabled: bool = True,            # set False at eval
        optimize: bool = True,           # set False at eval
        seed: int | None = None,
        in_channels: int = 3,            # 3 for RGB; if you framestack (12), set 12 here
    ):
        super().__init__(env)
        self.n_actions = int(n_actions)
        self.total_steps = int(total_steps)
        self.beta = float(beta)
        self.cap = float(cap_per_ep)
        self.decay_start = float(decay_start)
        self.decay_stop  = float(decay_stop)
        self.use_inverse = bool(use_inverse)
        self.lr = float(lr)
        self.max_grad_norm = float(max_grad_norm)
        self.emb_dim = int(emb_dim)
        self.ebar_clip = float(ebar_clip)
        self.enabled = bool(enabled)
        self.optimize_icm = bool(optimize)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        # ICM modules (separate from DQN!)
        self.encoder = IcmEncoder(in_ch=in_channels, emb_dim=self.emb_dim).to(self.device)
        self.fwd     = IcmForward(emb_dim=self.emb_dim, n_actions=self.n_actions).to(self.device)
        self.inv     = IcmInverse(emb_dim=self.emb_dim, n_actions=self.n_actions).to(self.device) if self.use_inverse else None

        params = list(self.encoder.parameters()) + list(self.fwd.parameters())
        if self.inv is not None:
            params += list(self.inv.parameters())
        self.opt = torch.optim.Adam(params, lr=self.lr)

        self.rms = RunningMeanStd()
        self._gstep = 0
        self._ep_cap_used = 0.0
        self._ep_bonus_total = 0.0  # track total bonus for this episode
        self._last_obs_np = None  # HWC uint8

    def _decay_scale(self):
        """Linear decay schedule: 1.0 until decay_start, then -> 0.0 by decay_stop."""
        p = min(1.0, self._gstep / max(1, self.total_steps))
        if p <= self.decay_start:
            return 1.0
        if p >= self.decay_stop:
            return 0.0
        t = (p - self.decay_start) / (self.decay_stop - self.decay_start)
        return 1.0 - t

    def reset(self, **kw):
        """Reset episode-level counters."""
        self._ep_cap_used = 0.0
        self._ep_bonus_total = 0.0
        obs, info = self.env.reset(**kw)
        self._last_obs_np = obs
        return obs, info

    def step(self, action: int):
        """Compute intrinsic reward and optimize ICM nets."""
        # Previous observation
        s_t_np = self._last_obs_np
        obs, r_ext, terminated, truncated, info = self.env.step(action)
        self._gstep += 1
        r_int = 0.0

        if self.enabled and self.optimize_icm and s_t_np is not None:
            # Prepare tensors
            s_t  = to_chw_float(s_t_np).to(self.device)
            s_tp1 = to_chw_float(obs).to(self.device)
            a_oh = one_hot(int(action), self.n_actions, self.device)

            # Forward pass
            z_t   = self.encoder(s_t)           # (1, emb)
            z_tp1 = self.encoder(s_tp1)         # share encoder (standard ICM)
            z_hat = self.fwd(z_t.detach(), a_oh)  # predict next embedding from (z_t, a)

            # Losses
            loss_fwd = F.mse_loss(z_hat, z_tp1.detach())  # predict z_{t+1}
            if self.inv is not None:
                logits_inv = self.inv(z_t, z_tp1)
                loss_inv = F.cross_entropy(logits_inv, torch.tensor([int(action)], device=self.device))
                loss = loss_fwd + 0.1 * loss_inv
            else:
                loss = loss_fwd

            # Optimize
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.fwd.parameters(), self.max_grad_norm)
            if self.inv is not None:
                nn.utils.clip_grad_norm_(self.inv.parameters(), self.max_grad_norm)
            self.opt.step()

            # Intrinsic reward (standardized, clipped, cap+decay)
            e_raw = float(loss_fwd.detach().item())  # scalar
            self.rms.update(e_raw)
            std = max(self.rms.std, 1e-6)
            e_bar = max(0.0, (e_raw - self.rms.mean) / std)  # ReLU to keep non-neg
            e_bar = min(e_bar, self.ebar_clip)
            scale = self._decay_scale()

            if scale > 0.0 and self._ep_cap_used < self.cap:
                r_int_step = self.beta * e_bar
                # respect remaining per-episode cap
                r_int_step = min(r_int_step, self.cap - self._ep_cap_used)
                r_int = float(scale * r_int_step)
                self._ep_cap_used += r_int_step
                self._ep_bonus_total += r_int

        # Total reward seen by DQN
        r_total = float(r_ext + r_int)

        # Update last obs and info (log)
        self._last_obs_np = obs
        info = dict(info) if isinstance(info, dict) else {}
        if self.enabled:
            info["icm_bonus_step"] = r_int
            info["icm_bonus_ep_sum"] = self._ep_bonus_total
            info["icm_progress"] = min(1.0, self._gstep / max(1, self.total_steps))
            info["icm_decay_scale"] = self._decay_scale()
            if r_ext != 0.0:
                info["icm_ratio_step"] = r_int / abs(r_ext)

        return obs, r_total, terminated, truncated, info

    def set_enabled(self, flag: bool):
        """Toggle intrinsic reward on/off."""
        self.enabled = bool(flag)

    def set_optimize(self, flag: bool):
        """Toggle ICM optimization on/off."""
        self.optimize_icm = bool(flag)
