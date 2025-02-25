import os
import random

import gymnasium as gym
import numpy as np
import torch

from sac import SACAgent
from utils import create_folder_if_not_exists, frames_to_gif, plot_learning_curve


class TrainingConfig:
    def __init__(
        self,
        seeds: list[int] = [1],
        environment: str = "InvertedPendulum-v5",
        episodes: int = 1000,
        save_path: str = "results",
        render_mode: str = "rgb_array",
        load_checkpoint: bool = False,
        debug_mode: bool = False,
    ):
        self.seeds = seeds
        self.episodes = episodes
        self.save_path = save_path
        self.render_mode = render_mode
        self.load_checkpoint = load_checkpoint
        self.environment = environment
        self.debug_mode = debug_mode

        self.check_environment()

    def check_environment(self):
        available_environments = [env.id for env in gym.envs.registry.values()]
        if self.environment not in available_environments:
            raise ValueError(
                f"Environment {self.environment} not found in Gymnasium registry.\nAvailable environments: {available_environments}"
            )

    def show(self):
        print("\nTraining Config:")
        print(f"Environment: {self.environment}")
        print(f"Seeds: {self.seeds}")
        print(f"Episodes: {self.episodes}")
        print(f"Save Path: {self.save_path}")
        print(f"Render Mode: {self.render_mode}")
        print(f"Load Checkpoint: {self.load_checkpoint}")


class TrainerHandler:
    def __init__(self, configs: list[TrainingConfig]):
        """
        :param configs: List of dictionaries with keys:
                        - "seed": int, random seed for reproducibility.
                        - "environment": str, the Gym environment name.
                        - "episodes": int, number of episodes to train.
                        - "save_path": str, directory path where plots (and possibly models) are saved.
                        Optional keys:
                        - "render_mode": str, Gym render mode (default "human").
                        - "load_checkpoint": bool, whether to load an existing checkpoint.
                        - "render_each_steps": int, frequency of calling env.render().
        """
        self.configs = configs

    def run_all_trainings(self):
        for config in self.configs:
            self.run_training(config)

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save_training_artifacts(self, frames, score_history, episode_info, paths):
        """
        # Save training artifacts like videos and learning curves
        Args:
            frames: List of frames from the episode
            score_history: Array of scores
            episode_info: Dict containing episode number, score and avg_score
            paths: Dict containing file paths for saving
        """
        video_saved = False
        learning_curve_saved = False

        try:
            # Save video if frames exist
            if frames and paths.get('gif_path'):
                create_folder_if_not_exists(os.path.dirname(paths['gif_path']))
                frames_to_gif(
                    frames,
                    paths['gif_path'],
                    metadata=episode_info,
                )
                video_saved = True
        except Exception as e:
            print(f"[Episode {episode_info['episode']}] Warning: Failed to save video - {str(e)}")

        try:
            # Save learning curve
            if paths.get('figure_file') and len(score_history) > 0:
                create_folder_if_not_exists(os.path.dirname(paths['figure_file']))
                x = [i + 1 for i in range(len(score_history))]
                plot_learning_curve(x, score_history, paths['figure_file'])
                learning_curve_saved = True
        except Exception as e:
            print(f"[Episode {episode_info['episode']}] Warning: Failed to save learning curve - {str(e)}")

        if not video_saved and not learning_curve_saved:
            print(f"[Episode {episode_info['episode']}] No artifacts to save\n")

        print(
            f"[Episode {episode_info['episode']}] Saved {'video' if video_saved else ''} {'and' if video_saved and learning_curve_saved else ''} {'learning curve' if learning_curve_saved else ''}\n"
        )

    def run_training(self, config: TrainingConfig):

        config.show()

        for seed in config.seeds:
            # Create the environment
            raw_env = gym.make(config.environment, render_mode=config.render_mode)

            env = gym.wrappers.RecordEpisodeStatistics(raw_env, 100)  # Records episode-reward

            # Set the seed for reproducibility
            self.set_seed(seed)

            # Create the base path for the training
            self.base_path = f"{config.save_path}/{config.environment}/seed_{seed}"
            create_folder_if_not_exists(self.base_path)

            # Create the learning curve and gif paths
            figure_file = f"{self.base_path}/learning_curve.png"
            gif_path = f"{self.base_path}/best_episode.gif"

            # Create the agent
            agent = SACAgent(
                input_dims=env.observation_space.shape,
                n_actions=env.action_space.shape[0],
                max_action=env.action_space.high,
                checkpoint_dir=f"{self.base_path}/sac_weights",
                alpha=0.0005,
                beta=0.0005,
                # tau=0.001,
                gamma=0.98,
                reward_scale=0.5,
                max_replay_buffer_size=config.episodes * 100,
                batch_size=512,
                layer1_size=256,
                layer2_size=256,
                debug_mode=config.debug_mode,
            )

            # Initialize the best score and score history
            best_score = np.float32(-np.inf)
            score_history = np.array([])

            if config.load_checkpoint:
                agent.load_models()

            # Train the agent for the given number of episodes
            for i in range(round(config.episodes)):
                # Reset the environment
                observation, info = env.reset(seed=seed)

                # Initialize the done flag, score and frames
                done = False
                score = np.float32(0)
                frames = []

                step = 0
                rewards_list = []
                # Train the agent until the episode is done
                while not done:
                    step += 1
                    # Save the frame to generate a video later
                    if config.render_mode == "rgb_array":
                        rgb_array = np.array(env.render())
                        frames.append(rgb_array)

                    # Choose the action to take based on the observation
                    action = agent.choose_action(observation)

                    # Take the action and get the next observation, reward, done flag and info
                    next_observation, reward, terminated, truncated, info = env.step(action)

                    # Append the reward to the rewards list
                    rewards_list.append(reward)

                    # Update the score
                    score += np.float32(reward)

                    # Update the done flag
                    done = terminated or truncated

                    # Define the times to remember - More than one for good experiences
                    # Here, if the reward is greater than the medium of the rewards, we remember it 3 times
                    times = 1 if reward < np.mean(rewards_list) else 3

                    # Remember the experience
                    agent.remember(observation, action, reward, next_observation, done, times=times)

                    # Learn if not loading a checkpoint
                    if not config.load_checkpoint:
                        agent.learn()

                    # Update the observation
                    observation = next_observation

                # Update the score history
                score_history = np.append(score_history, score)

                # Calculate the average score of the last 100 episodes
                avg_score = np.mean(score_history[-100:])

                print(f"[Episode {i}] Score: {score:.2f} Avg Score: {avg_score:.2f}")

                # Update the best score if the average score is higher
                if avg_score > best_score:
                    best_score = avg_score

                    # Save the models if not loading a checkpoint
                    if not config.load_checkpoint:
                        saved = agent.save_models()

                        if not saved:
                            print(f"[Episode {i}] Warning: Failed to save models\n")
                        else:

                            episode_info = {'episode': i, 'score': score, 'avg_score': avg_score}
                            wrapper_info = info['episode']

                            episode_info = {**episode_info, **wrapper_info}
                            print(f"[Episode {i}] Saving models. Episode info: {episode_info}")

                            # Save artifacts with error handling
                            self.save_training_artifacts(
                                frames if config.render_mode == "rgb_array" else None,
                                score_history[: i + 1],
                                episode_info,
                                {
                                    'gif_path': gif_path,
                                    'figure_file': figure_file,
                                },
                            )

            # Save final learning curve
            if not config.load_checkpoint:
                self.save_training_artifacts(
                    None,
                    score_history,
                    {'episode': config.episodes, 'score': score, 'avg_score': avg_score},
                    {'figure_file': f"{self.base_path}/final_learning_curve.png"},
                )

            # Close the environment
            env.close()
            print(f"Closed environment for seed {seed}\n")

    def run_evaluation(self, config: TrainingConfig):
        """
        Evaluates a trained agent using the provided configuration
        """
        # Create environment
        raw_env = gym.make(config.environment, render_mode=config.render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(raw_env, 100)

        # Create agent and load trained model
        agent = SACAgent(
            input_dims=env.observation_space.shape,
            n_actions=env.action_space.shape[0],
            max_action=env.action_space.high,
            checkpoint_dir=f"{config.save_path}/{config.environment}/seed_{config.seeds[0]}/sac_weights",
        )
        agent.load_models()

        score_history = []

        # Evaluate the agent
        for i in range(config.episodes):
            observation, info = env.reset()
            done = False
            score = np.float32(0)
            episode_frames = []

            while not done:
                if config.render_mode == "rgb_array":
                    rgb_array = np.array(env.render())
                    episode_frames.append(rgb_array)

                action = agent.choose_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                score += np.float32(reward)
                done = terminated or truncated

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            print(f"[Episode {i}] Score: {score:.2f} Avg Score: {avg_score:.2f}")

            # Save GIF for best episode
            if score == max(score_history):
                if config.render_mode == "rgb_array":
                    gif_path = f"{config.save_path}/{config.environment}/evaluation.gif"
                    self.save_training_artifacts(
                        episode_frames,
                        score_history,
                        {'episode': i, 'score': score, 'avg_score': avg_score},
                        {'gif_path': gif_path},
                    )

        env.close()

    def run_all_evaluations(self):
        for config in self.configs:
            self.run_evaluation(config)
