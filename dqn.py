import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, hidden_dim=512, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        action_dim = 2
        self.enable_dueling_dqn = enable_dueling_dqn

        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=8, stride=4
        )  # [N, 1, 64, 64] -> [N, 32, 15, 15]
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2
        )  # [N, 32, 15, 15] -> [N, 64, 6, 6]

        # Calculate the size of the flattened tensor after convolutional layers
        self.conv_out_dim = 64 * 6 * 6  # Flattened dimension after conv layers

        # Fully connected layers
        self.fc1 = nn.Linear(self.conv_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 128)
            self.value = nn.Linear(128, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 128)
            self.advantages = nn.Linear(128, action_dim)
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Reshape input for convolutional layers
        x = x.view(x.size(0), 1, 64, 64)  # [batch_size, 1, 64, 64]

        # Pass through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:
            # Value calculation
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calculation
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Q-value calculation
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q
