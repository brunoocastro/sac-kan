import gymnasium as gym
import numpy as np

from sac import SACAgent
from utils import create_folder_if_not_exists, plot_learning_curve

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    agent = SACAgent(
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0],
        max_action=env.action_space.high,
    )

    n_games = 100
    filename = "inverted_pendulum.png"

    figure_file = "plots/" + filename

    best_score = float("-inf")  # Inicializa com o menor valor possível
    score_history = []
    load_checkpoint = False

    render_each_steps = 100

    if load_checkpoint:
        agent.load_models()
        env.render()

    for i in range(n_games + 1):
        observation, info = env.reset()
        print(observation, info)
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
            agent.remember(observation, action, reward, next_observation, done)
            if not load_checkpoint:
                agent.learn()
            observation = next_observation

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print(f"[Episode {i}] Score: {score:.2f} Avg Score: {avg_score:.2f}")

        if i % render_each_steps == 0:
            env.render()

    if not load_checkpoint:
        create_folder_if_not_exists(figure_file)
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)

    env.close()
