"""
Neural Network Architectures for RL Agents
Includes Q-Networks (DQN) and Actor-Critic (PPO)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Categorical

class QNetwork(nn.Module):
    """Deeep Q network for processing visual observations"""

    def __init__(self, observation_shape: Tuple[int,int,int] = (3,64,64), num_actions: int = 17, hidden_dim: int = 256):
        super(QNetwork, self).__init__()

        # Layer 1: 3x3 conv with 32 filters, stride 2 for downsampling
        # Input: (3, 64, 64) -> Output: (32, 31, 31)
        self.conv1 = nn.Conv2d(3,32,kernel_size=3, stride = 2, padding = 0)

        # Layer 2: 3x3 conv with 64 filters, stride 2
        # Input: (32, 31, 31) -> Output: (64, 15, 15)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride = 2, padding = 0)

        # Layer 3: 3x3 conv with 128 filters, stride 2
        # Input: (64, 15, 15) -> Output: (128, 7, 7)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)

        # Layer 4: 3x3 conv with 256 filters, stride 2
        # Input: (128, 7, 7) -> Output: (256, 3, 3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0)

        # Calculate flattened size after convolutions
        # (256 channels * 3 height * 3 width = 2304)
        self.flatten_size = 256 * 3 * 3

        # MLP Head for Q-values
        # Two hidden layers with ReLU activation
        self.fc1 = nn.Linear(self.flatten_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer: Q-value for each action
        self.q_values = nn.Linear(hidden_dim, num_actions)

        # Initialize weights using He initialization for ReLU networks
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        Proper initialization is crucial for training deep networks.
        He initialization works well with ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out',nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # He initialization for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out',nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs:torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            obs: Batch of observations, shape (batch_size, 3, 64, 64)
                Values should be in [0, 1] range

        Returns:
            Q-values for each action, shape (batch_size, num_actions)
        """
        # CNN Encoder with ReLU activations
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten for MLP
        # Use contiguous() to ensure tensor is in contiguous memory before view
        x = x.contiguous().view(x.size(0), -1)  # (batch_size, flatten_size)

        # MLP Head
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output Q-values (no activation - Q-values can be any real number)
        q_values = self.q_values(x)

        return q_values
    
    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """
        Convenience method to get Q-values from numpy observations.

        Handles the numpy -> torch -> numpy conversion.

        Args:
            obs: Numpy array of observations (batch_size, 64, 64, 3) or (64, 64, 3)

        Returns:
            Q-values as numpy array
        """
        # Handle single observation
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]  # Add batch dimension

        # Convert from HWC to CHW format for PyTorch
        obs = obs.transpose(0, 3, 1, 2)  # (batch, 64, 64, 3) -> (batch, 3, 64, 64)

        # Convert to torch tensor
        obs_torch = torch.FloatTensor(obs).to(next(self.parameters()).device)

        # Get Q-values
        with torch.no_grad():
            q_values = self.forward(obs_torch)

        return q_values.cpu().numpy()
    
class ImprovedQNetwork(QNetwork):
    """
    Enhanced Q-Network for Improvement 1.
    
    Adds batch normalization and larger architecture for better performance.
    This will be used when we add data augmentation.
    """

    def __init__(self, *args, use_batch_norm: bool = True, **kwargs):
        """
        Initialize improved network with optional batch normalization.
        
        Args:
            use_batch_norm: Whether to use batch normalization layers
        """
        self.use_batch_norm = use_batch_norm
        super().__init__(*args, **kwargs)

        if self.use_batch_norm:
            # Add batch norm layers after each conv layer
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with batch normalization.

        Batch norm helps with training stability and allows higher learning rates.
        """
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
    """
    Create a target network by copying a source network.

    Target networks provide stable Q-value targets during training.
    They are updated slowly to prevent oscillations.

    Args:
        source_network: Network to copy

    Returns:
        Deep copy of the source network
    """
    import copy
    target = copy.deepcopy(source_network)

    # Freeze target network (we'll update it manually)
    for param in target.parameters():
        param.requires_grad = False

    return target

def soft_update(source_network: nn.Module,target_network: nn.Module,tau: float = 0.01) -> None:
    """
    Soft update of target network parameters.

    Instead of copying weights directly, we blend them:
    target = tau * source + (1 - tau) * target

    This provides more stable learning than hard updates.

    Args:
        source_network: Network with latest weights
        target_network: Network to update slowly
        tau: Soft update coefficient (0.01 = 1% new, 99% old)
    """
    for source_param, target_param in zip(source_network.parameters(),
                                        target_network.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

# Test the network shapes
if __name__ == "__main__":
    """
    Test script to verify network dimensions.
    Run this to make sure the architecture works correctly.
    """
    # Create a test network
    net = QNetwork(observation_shape=(3, 64, 64), num_actions=17)

    # Test with random input
    test_input = torch.randn(32, 3, 64, 64)  # Batch of 32 images
    output = net(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (32, 17)")

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")


# ============================================================================
# PPO Actor-Critic Network
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with shared CNN backbone.

    Architecture:
        - Shared CNN: Extracts visual features from 64x64 RGB observations
        - Actor head: Maps features to action probabilities (policy π(a|s))
        - Critic head: Maps features to state value V(s)

    The shared backbone allows both policy and value to learn from the same
    visual representations, improving sample efficiency.
    """

    def __init__(self, num_actions: int = 17, hidden_dim: int = 512):
        """
        Initialize Actor-Critic network.

        Args:
            num_actions: Number of discrete actions (17 for Crafter)
            hidden_dim: Hidden dimension for policy/value heads (default 512)

        Note: Input shape is fixed to (3, 64, 64) for Crafter environment
        """
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions

        # Shared CNN backbone (same architecture as Q-network for consistency)
        # Processes 64x64 RGB images into feature vectors
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)   # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8

        # Calculate flattened feature size: 64 channels * 8 * 8 = 4096
        self.feature_size = 64 * 8 * 8

        # Actor head (policy network)
        # Maps features to action logits (unnormalized log probabilities)
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Critic head (value network)
        # Maps features to scalar state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using orthogonal initialization.

        Orthogonal initialization is recommended for PPO as it helps with
        gradient flow and prevents vanishing/exploding gradients.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            obs: Batch of observations, shape (batch_size, 3, 64, 64)
                 Values should be in [0, 1] range

        Returns:
            action_logits: Unnormalized log probabilities, shape (batch_size, num_actions)
            values: State values V(s), shape (batch_size, 1)
        """
        # Shared CNN feature extraction
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten features
        x = x.contiguous().view(x.size(0), -1)  # (batch_size, feature_size)

        # Actor: compute action logits
        action_logits = self.actor(x)  # (batch_size, num_actions)

        # Critic: compute state value
        values = self.critic(x)  # (batch_size, 1)

        return action_logits, values

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            obs: Observation tensor, shape (batch_size, 3, 64, 64) or (3, 64, 64)
            deterministic: If True, return argmax action (for evaluation)
                          If False, sample from policy distribution (for training)

        Returns:
            action: Selected action, shape (batch_size,)
            log_prob: Log probability of the action, shape (batch_size,)
            value: State value V(s), shape (batch_size, 1)
        """
        # Add batch dimension if needed
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        # Forward pass
        action_logits, values = self.forward(obs)

        # Create categorical distribution from logits
        dist = Categorical(logits=action_logits)

        if deterministic:
            # Greedy action (for evaluation)
            action = action_logits.argmax(dim=-1)
        else:
            # Sample action from policy (for training/exploration)
            action = dist.sample()

        # Compute log probability of the action
        log_prob = dist.log_prob(action)

        return action, log_prob, values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions taken in given states (used during PPO update).

        This is the key method for computing the PPO loss. It re-evaluates
        old actions under the current policy to compute the probability ratio.

        Args:
            obs: Batch of observations, shape (batch_size, 3, 64, 64)
            actions: Batch of actions taken, shape (batch_size,)

        Returns:
            log_probs: Log probabilities of actions under current policy, shape (batch_size,)
            values: State values V(s), shape (batch_size, 1)
            entropy: Entropy of the policy distribution, shape (batch_size,)
        """
        # Forward pass
        action_logits, values = self.forward(obs)

        # Create distribution
        dist = Categorical(logits=action_logits)

        # Evaluate log probability of the given actions
        log_probs = dist.log_prob(actions)

        # Compute entropy (measure of policy randomness)
        # Higher entropy = more exploration
        entropy = dist.entropy()

        return log_probs, values, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value V(s) without sampling action.

        Useful for computing advantages during rollout collection.

        Args:
            obs: Observation tensor, shape (batch_size, 3, 64, 64) or (3, 64, 64)

        Returns:
            values: State values, shape (batch_size, 1) or (1,)
        """
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)

        _, values = self.forward(obs)
        return values