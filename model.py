import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"
)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
        self,
        state_size,
        action_size,
        seed,
        hidden_layer_sizes=[512, 256, 128],
        dropout_prob=0.25,
    ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        # For the inputs we probably want some history, but not sure exactly how to
        # enable that at the moment. The observation space already includes velocity,
        # so maybe we don't need to worry about this.

        # For the lander, the state space is:
        # x, y, xdot, ydot, theta, theta_dot, left_contact, right_contact

        # Let's just try throwing a bunch of fully connected linear layers at the
        # problem and see what we come out with!

        assert len(hidden_layer_sizes) > 0, "Expected at least 1 hidden layer"

        # Input layer
        self.input = torch.nn.Linear(state_size, hidden_layer_sizes[0])

        hidden_layers = []
        dropout_layers = []
        for in_, out_ in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]):
            hidden_layers.append(torch.nn.Linear(in_, out_))
            dropout_layers.append(torch.nn.Dropout(dropout_prob))

        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.dropout_layers = nn.ModuleList(dropout_layers)

        # Here we need an output for each action
        self.output = torch.nn.Linear(hidden_layer_sizes[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input(state)
        x = F.relu(x)
        for linear, dropout in zip(self.hidden_layers, self.dropout_layers):
            x = linear(x)
            x = F.relu(x)
            # Don't apply dropout layers for now - add this later
            x = dropout(x)

        x = self.output(x)

        return x
