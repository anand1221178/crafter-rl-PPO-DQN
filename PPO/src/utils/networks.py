
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Categorical

class QNetwork(nn.Module):
    #TODO: add comments explaining netowrks - done

    def __init__(self, observation_shape: Tuple[int,int,int] = (3,64,64), num_actions: int = 17, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        # Layer 1 -> 3x3 conv with 32 filters stride 2 for downsampling
        # Input ->  (3, 64, 64) -> (32, 31, 31)
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride = 2, padding = 0)

        # Layer 2-  3x3 conv with 64 filters stride 2
        # Input -> (32, 31, 31) ->  (64, 15, 15)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride = 2, padding = 0)

        # Layer 3-  3x3 conv with 128 filters  stride 2
        # Input -> (64, 15, 15) -> (128, 7, 7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)

        # Layer 4 -  3x3 conv with 256 filters stride 2
        # Input ->  (128, 7, 7) ->(256, 3, 3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)


        self.flatten_size = 256 * 3 * 3


        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer
        self.q_values = nn.Linear(hidden_dim, num_actions)

        # Irelu
        self._initialize_weights()
    
    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(module.weight, mode='fan_out',nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(module.weight, mode='fan_out',nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        # CNN Encoder with ReLU activations
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten for MLP
        x = x.contiguous().view(x.size(0), -1) 

        # MLP Head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output Qvals
        q_values = self.q_values(x)

        return q_values
    
    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        #single obs
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...] 


        obs = obs.transpose(0, 3, 1, 2) 

    
        obs_torch = torch.FloatTensor(obs).to(next(self.parameters()).device)


        with torch.no_grad():
            q_values = self.forward(obs_torch)

        return q_values.cpu().numpy()
    
class ImprovedQNetwork(QNetwork):

    def __init__(self, *args, use_batch_norm: bool = True, **kwargs):

        self.use_batch_norm = use_batch_norm
        super().__init__(*args, **kwargs)

        if self.use_batch_norm:
            # Add batch norm layers after each conv layer
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        # CNN Encoder with batch norm
        x = self.conv1(obs)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        if self.use_batch_norm:
            x = self.bn4(x)
        x = F.relu(x)

        # Flatten and MLP head (same as base network)
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.q_values(x)

        return q_values
    
def create_target_network(source_network: nn.Module) -> nn.Module:

    import copy
    target = copy.deepcopy(source_network)

    # Freeze target network
    for param in target.parameters():
        param.requires_grad = False

    return target

def soft_update(source_network: nn.Module,target_network: nn.Module,tau: float = 0.01) -> None:

    for source_param, target_param in zip(source_network.parameters(),target_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

# Test the network shapes
if __name__ == "__main__":

    net = QNetwork(observation_shape=(3, 64, 64), num_actions=17)


    test_input = torch.randn(32, 3, 64, 64)  # Batch of 32 images
    output = net(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (32, 17)")


    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")


# ==============================================================================
# PPO  Network
# ========================================================================

class ActorCritic(nn.Module):

    def __init__(self, num_actions: int = 17, hidden_dim: int = 512):

        super(ActorCritic, self).__init__()

        self.num_actions = num_actions

        # Shared CNN 
        # Processes 64 by 64 RGB images into feature vectors
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)   # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8

        # Calculate flattened feature size: 64 channels times 8 times 8 = 4096
        self.feature_size = 64 * 8 * 8

        # Actor head policy net
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Critic head value net
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


        self._initialize_weights()

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Shared CNN feature extraction 
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten features
        x = x.contiguous().view(x.size(0), -1)  


        action_logits = self.actor(x) 

       
        values = self.critic(x) 

        return action_logits, values

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        # Forward pass
        action_logits, values = self.forward(obs)

        # Create categorical distribution 
        dist = Categorical(logits=action_logits)

        if deterministic:
            # Greedy action for eval
            action = action_logits.argmax(dim=-1)
        else:
            # Sample action from policy (for training
            action = dist.sample()

        #  log probability of the action
        log_prob = dist.log_prob(action)

        return action, log_prob, values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Forward pass
        action_logits, values = self.forward(obs)

        # Create distribution
        dist = Categorical(logits=action_logits)

        #  log probability of the given actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        _, values = self.forward(obs)
        return values