import numpy as np
import torch as T
import torch.nn.functional as F

from sac.buffer import ReplayBuffer
from sac.networks import ActorNetwork, CriticNetwork, ValueNetwork


class SACAgent:
    """
    SAC (Soft Actor-Critic) is a model-free off-policy algorithm for continuous action spaces.
    It combines the advantages of actor-critic methods and Monte Carlo tree search (MCTS) to find the optimal policy.

    The algorithm consists of:
    - An actor network that generates actions from states
    - A critic network that evaluates the quality of the actions
    - A value network that estimates the expected future rewards
    - A target network that is used to compute the target values
    - A replay buffer that stores the experiences
    - A soft update rule that updates the target network
    - A reward scaling factor that scales the rewards
    - A batch size that determines the number of experiences to sample from the replay buffer
    - A learning rate that determines the step size of the optimizer


    The SAC algorithm is designed to:
    - Optimize the policy by maximizing the expected future rewards
    - Ensure that the policy is smooth and continuous
    - Handle the exploration-exploitation trade-off
    - Provide a probabilistic policy that can be sampled from
    - Allow for efficient off-policy training

    The parameters are:
    - max_action: The maximum action value
    - alpha: The learning rate for the actor network
    - beta: The learning rate for the critic network
    - input_dims: The dimensions of the input state
    - gamma: The discount factor
    - n_actions: The number of actions
    - max_replay_buffer_size: The maximum size of the replay buffer
    - tau: The soft update parameter
    - layer1_size: The size of the first layer of the actor and critic networks
    - layer2_size: The size of the second layer of the actor and critic networks
    - batch_size: The number of experiences to sample from the replay buffer
    - reward_scale: The scaling factor for the rewards


    """

    def __init__(
        self,
        max_action,
        alpha=0.0003,
        beta=0.0003,
        input_dims=[8],
        gamma=0.99,
        n_actions=2,
        max_replay_buffer_size=1000000,
        tau=0.005,
        layer1_size=256,
        layer2_size=256,
        batch_size=256,
        reward_scale=2,
        checkpoint_dir="tmp/sac",
        debug_mode=False,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size=max_replay_buffer_size, input_shape=input_dims, n_actions=n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.debug_mode = debug_mode

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        print(f"SAC factored using Device: {self.device}")

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

    def remember(self, state, action, reward, new_state, done, times=1):
        """
        Stores the experience in the replay buffer.

        Parameters:
            state: np.ndarray
                The current state of the environment.
            action: np.ndarray
                The action taken in the current state.
            reward: float
                The reward received for taking the action.
            new_state: np.ndarray
                The new state of the environment after taking the action.
            done: bool
                A boolean flag indicating if the episode is over.
            times: int
                The number of times to store the experience in the replay buffer. Can be used to store multiple
                experiences at once. Defaults to 1. Could be useful for storing multiple good experiences at once.
        """
        for _ in range(times):
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
        if self.memory.mem_cntr < self.batch_size:
            if self.debug_mode:
                print(
                    f"Skipping saving models due to insufficient memory. Memory size: {self.memory.mem_cntr}, Batch size: {self.batch_size}"
                )
            return False

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
            if self.debug_mode:
                print(
                    f"Skipping learning step due to insufficient memory. Memory size: {self.memory.mem_cntr}, Batch size: {self.batch_size}"
                )
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Transform into Pytorch Tensors - Do here to avoid ReplayBuffer to be linked to Pytorch, staying framework agnostic
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)  # all network devices is the same
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        # If any of tensors are "nan", stop the entire process to debug
        if (
            state.isnan().any()
            or action.isnan().any()
            or reward.isnan().any()
            or state_.isnan().any()
            or done.isnan().any()  # noqa
        ):
            print("[LEARN] Found NaN values on tensors. Stopping the process to debug.")
            print(f"[LEARN] state: {state}")
            print(f"[LEARN] action: {action}")
            print(f"[LEARN] reward: {reward}")
            print(f"[LEARN] state_: {state_}")
            print(f"[LEARN] done: {done}")
            exit(1)

        # Calculate the values of the states and new_states according to the valueNetwork and targetValueNetwork
        value = self.value(state).view(-1)  # Claps as batch_dim
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0  # Definition of the value function

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)

        if self.debug_mode:
            # print(f"[LEARN] Actions: {actions}")
            print(f"[LEARN] Probs: {log_probs}")

        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)

        # The real critic value is the minimum between the two critic networks that we have
        # We do this bc improves the stability of learning (solving q-learning overestimation bias problem) - TD3 paper demonstrates this
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)  # Claps batch_dim

        if q1_new_policy.isnan().any() or q2_new_policy.isnan().any() or critic_value.isnan().any():
            print("[LEARN] Found NaN values on critic values. Stopping the process to debug.")
            print(f"[LEARN] q1_new_policy: {q1_new_policy}")
            print(f"[LEARN] q2_new_policy: {q2_new_policy}")
            print(f"[LEARN] critic_value: {critic_value}")
            exit(1)

        # Calculate Value Network loss and backpropagate
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        if value.isnan().any() or value_target.isnan().any() or value_loss.isnan().any():
            print("[LEARN] Found NaN values on value loss. Stopping the process to debug.")
            print(f"[LEARN] value: {value}")
            print(f"[LEARN] value_target: {value_target}")
            print(f"[LEARN] value_loss: {value_loss}")
            exit(1)

        # Calculate Actor Network loss and backpropagate
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, action)
        q2_new_policy = self.critic_2.forward(state, action)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)  # Claps batch_dim

        if log_probs.isnan().any() or critic_value.isnan().any():
            print("[LEARN] Found NaN values on actor values. Stopping the process to debug.")
            print(f"[LEARN] log_probs: {log_probs}")
            print(f"[LEARN] critic_value: {critic_value}")
            exit

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        if actor_loss.isnan().any():
            print("[LEARN] Found NaN values on actor loss. Stopping the process to debug.")
            print(f"[LEARN] actor_loss: {actor_loss}")
            exit(1)

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

        if critic1_loss.isnan().any() or critic2_loss.isnan().any() or critic_loss.isnan().any():
            print("[LEARN] Found NaN values on critic loss. Stopping the process to debug.")
            print(f"[LEARN] critic1_loss: {critic1_loss}")
            print(f"[LEARN] critic2_loss: {critic2_loss}")
            print(f"[LEARN] critic_loss: {critic_loss}")
            exit(1)

        if self.debug_mode:
            print(f"[LEARN] value_loss: {value_loss}")
            print(f"[LEARN] value_target: {value_target}")
            print(f"[LEARN] critic_value: {critic_value}")
            print(f"[LEARN] critic1_loss: {critic1_loss}")
            print(f"[LEARN] critic2_loss: {critic2_loss}")
            print(f"[LEARN] critic_loss: {critic_loss}")
            print(f"[LEARN] actor_loss: {actor_loss}")
            print(f"[LEARN] q_hat: {q_hat}")
            print(f"[LEARN] q1_old_policy: {q1_old_policy}")
            print(f"[LEARN] q2_old_policy: {q2_old_policy}")

        self.update_network_parameters()
