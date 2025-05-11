import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QNetwork
from replay_buffer import PrioritizedReplayBuffer


# Default PER Hyperparameters
_PER_ALPHA = 0.6  # prioritization exponent (0=uniform, 1=full)
_PER_BETA = 0.4  # initial importance sampling exponent
_PER_BETA_INCREMENT = 0  # beta annealing factor per sample step
_PER_EPSILON = 1e-4  # small value added to priorities to ensure non-zero probability

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
)

if device.type == "cude":
    device = torch.cuda.current_device()
    print(torch.cuda.get_device_name(device))
elif device.type == "cpu":
    print("No accelerators found. Using CPU")

class Agent:
    """Interacts with and learns from the environment."""

    replay_buffer_size = int(1e5)
    
    batch_size = 64  # minibatch size
    gamma = 0.99 # discount factor
    tau = 1e-3 # soft update of target parameters
    learning_rate = 4e-4

    target_network_update_interval = 5

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        hidden_layer_sizes=[256, 128, 64],
        dropout_prob=0.25,
        use_double_dqn=False,
        per_epsilon=_PER_EPSILON,
        per_alpha=_PER_ALPHA,
        per_beta=_PER_BETA,
        per_beta_increment=_PER_BETA_INCREMENT,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            per_alpha (float): Alpha parameter for prioritized experience replay (0 = standard replay, 1 = full prioritization)
            per_beta (float): Beta parameter for prioritized experience replay (initial value)
            per_beta_increment (float): Beta increment per sample step
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_double_dqn = use_double_dqn
        self.per_epsilon = per_epsilon

        # Q-Network
        self.qnetwork_local = QNetwork(
            state_size, action_size, seed, hidden_layer_sizes, dropout_prob
        ).to(device)
        self.qnetwork_target = QNetwork(
            state_size, action_size, seed, hidden_layer_sizes, dropout_prob
        ).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(
            state_size,
            action_size,
            self.replay_buffer_size,
            self.batch_size,
            seed,
            device=device,
        )  # Pass device
        self.memory.alpha = per_alpha
        self.memory.beta = per_beta
        self.memory.beta_increment_per_sampling = per_beta_increment
        self.memory.epsilon = per_epsilon

        # Initialize time step (for updating every 'target_network_update_interval' steps)
        self.t_step = 0

    def train(self):
        self.qnetwork_local.train()
        self.qnetwork_target.train()

    def eval(self):
        self.qnetwork_local.eval()
        self.qnetwork_target.eval()

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.target_network_update_interval
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # Sample with priorities, get experiences, weights, and indices
                experiences, weights, indices = self.memory.sample()
                self.learn(experiences, self.gamma, weights, indices)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, weights=None, indices=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            weights (torch.Tensor): importance sampling weights (for PER)
            indices (list): list of indices of sampled experiences (for PER)
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute the Q-values for the given states and selected actions based on the
        # current network parameters
        q_values = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            if self.use_double_dqn:
                # For Double DQN, we need to select the action with the highest Q-value from our
                # local model, and then use that action to get the Q-value from the target model
                _, best_actions = self.qnetwork_local(next_states).detach().max(1, keepdim=True)
                # Use the best action to get the Q-value from the target model
                # This is the Double DQN trick
                next_q_values = self.qnetwork_target(next_states).detach().gather(1, best_actions)
            else:
                # For standard DQN, we just use the target model to get the Q-values (based on the
                # optimal actions) for the next states
                next_q_values = self.qnetwork_target(next_states).detach().max(1, keepdim=True)[0]

            # Compute Q targets for current states
            target_q_values = rewards + (gamma * next_q_values * (1 - dones))

        # Compute loss

        # Calculate element-wise loss so we can apply the weights
        elementwise_loss = F.mse_loss(q_values, target_q_values, reduction="none")
        # Apply importance sampling weights (weights are 1.0 if alpha=0)
        loss = (weights * elementwise_loss).mean()

        # Update priorities in the buffer (if using PER)
        if self.memory.alpha > 0:
            # Calculate absolute TD errors |Q_targets - Q_expected|
            td_errors = (target_q_values - q_values).abs().detach().cpu().numpy()
            # Add epsilon and update priorities
            new_priorities = td_errors + self.per_epsilon
            self.memory.update_priorities(indices, new_priorities.flatten())

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
