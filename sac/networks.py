import os

import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import action
from torch.distributions.normal import Normal
import numpy as np

def create_folder_if_not_exists(folder):
    if not os.path.exists(folder):
        print(f"Creating folder {folder} to save the rewards.")
        os.makedirs(folder)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name="critic", checkpoint_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Critic evaluates the state/action pair, so we add actions as input early - Review
        self.fc1 = nn.Linear(self.input_dims[0] + self.n_actions, self.fc1_dims)
        # Second layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Output layer - Defines the quality of the action
        self.q = nn.Linear(self.fc2_dims, 1)

        # Optimize our parameters based on deviation (self.parameters of nn.Module base class) using defined beta (Learning Rate)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Take usage of a GPU if available
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Send our network to GPU or CPU - depending on availability
        self.to(self.device)

    def forward(self, state, action):

        # First layer feedforward and activation
        ## Action value starts being the nn feedforward of the concatenation of state and value (as we defined above)
        ## Along the batch dimension (dim=1 - second one = fc1_dims)
        action_value = self.fc1(T.cat([state, action], dim=1))
        ## Than the activation of the action value
        action_value = F.relu(action_value)

        # Second layer feedforward and activation
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        # Q layer - feedforward the Q Value
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        create_folder_if_not_exists(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    """
    Value Network based on the SAC original article
    2 layer deep network ??

    The value network should receive a specific state and estimate the value of that state.
    So, with these goal, we receive a action, pass is through a 2 layer network and then flat it into a value, that is the estimation.

    Notes:
        - We don't need the number of actions here because the ValueNetwork just estimates the value for a specific state (or a set of states) so it doesn't care about the actions (took or are taken).
    """
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256, name='value', checkpoint_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Defines the layers of the network as Linear layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        # First layer feedforward and activation
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        # Second layer feedforward and activation
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        # Feedforward the value of the estimated value
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        create_folder_if_not_exists(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor', checkpoint_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # Reparameterization noise - substitute log(0) that is undefined
        self.reparam_noise = 1e-6

        # Define the Deep Neural Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Results in two outputs:
        ## MU = Mean of the distribution for a policy
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        ## SIGMA = Standard deviation
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):

        # Feedforward for the layers of the network
        probability = self.fc1(state)
        probability = F.relu(probability)
        probability = self.fc2(probability)
        probability = F.relu(probability)

        mu = self.mu(probability)
        sigma = self.sigma(probability)

        # Clamp sigma to not reach any values, maintaining it inside a very small number (higher than zero bc of Pytorch) and 1
        # This step is a faster way to do a sigmoid activation of the sigma output
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            # rsample - sample with some noise
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(self.device)

        # Calculation of loss function - for updating the weights of the nn
        log_probability = probabilities.log_prob(actions)
        ## Article appendix - Handle the scaling of the action (tanh)
        log_probability -= T.log(1-action.pow(2)+self.reparam_noise)
        ## Pytorch framework needs to return a scalar quantity for the loss
        log_probability = log_probability.sum(1, keepdim=True)

        return action, log_probability

    def save_checkpoint(self):
        create_folder_if_not_exists(self.checkpoint_dir)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))