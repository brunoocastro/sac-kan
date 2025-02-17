import numpy as np
import torch as T
import torch.nn.functional as F
from agents.new_sac.buffer import ReplayBuffer
from agents.new_sac.networks import ActorNetwork, CriticNetwork, ValueNetwork


class SACAgent:
    def __init__(
        self,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[8],
        # env=None,
        gamma=0.99,
        n_actions=2,
        max_replay_buffer_size=1000000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
        max_action=None,
        checkpoint_dir="tmp/sac",
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size=max_replay_buffer_size, input_shape=input_dims, n_actions=n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            n_actions=n_actions,
            name="actor",
            max_action=max_action,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            checkpoint_dir=checkpoint_dir,
        )
        self.critic_1 = CriticNetwork(
            beta,
            input_dims,
            n_actions=n_actions,
            name="critic_1",
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            checkpoint_dir=checkpoint_dir,
        )
        self.critic_2 = CriticNetwork(
            beta,
            input_dims,
            n_actions=n_actions,
            name="critic_2",
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            checkpoint_dir=checkpoint_dir,
        )
        self.value = ValueNetwork(
            beta,
            input_dims,
            name="value",
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            checkpoint_dir=checkpoint_dir,
        )
        self.target_value = ValueNetwork(
            beta,
            input_dims,
            name="target_value",
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            checkpoint_dir=checkpoint_dir,
        )

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor(np.array([observation])).to(self.actor.device)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        """
        Updates the network parameters using the soft update rule. The parameters of the main network are updated as a
        weighted combination of its current weights and the corresponding weights of the target network. This allows
        smooth parameter updates between the two networks.

        So, in the start, the target network will have the same weights as the main network. As the training progresses,
        we want to do a soft-copy, based on TAU value. If TAU is equals 1, the target network will be hard-copied form
        the value network. In any other case, it will be a "soft" copy, mixing the values.

        Parameters:
            tau: Optional[float]
                The soft update parameter controlling the influence of the target network's weights during the update.
                Must be a float value between 0 and 1. If not provided, a default value will be used.
        """
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + (1 - tau) * target_value_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print(".... Saving models ....")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        # Validate if we have at least the batch size on the memory, otherwise cancel learning step.
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Transform into Pytorch Tensors - Do here to avoid ReplayBuffer to be linked to Pytorch, staying framework agnostic
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)  # all network devices is the same
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        # Calculate the values of the states and new_states according to the valueNetwork and targetValueNetwork
        value = self.value(state).view(-1)  # Claps as batch_dim
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0  # Definition of the value function

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)

        # The real critic value is the minimum between the two critic networks that we have
        # We do this bc improves the stability of learning (solving q-learning overestimation bias problem) - TD3 paper demonstrates this
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)  # Claps batch_dim

        # Calculate Value Network loss and backpropagate
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        # Calculate Actor Network loss and backpropagate
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)  # Claps batch_dim

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Deal with critic loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
