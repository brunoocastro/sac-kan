import gymnasium as gym
import numpy as np

from sac import SACAgent
from utils import create_folder_if_not_exists, frames_to_gif, plot_learning_curve

if __name__ == "__main__":
    env_name = "InvertedPendulum-v5"
    total_episodes = 10000  # Reduzindo o número de episódios já que só queremos ver o resultado
    load_checkpoint = True  # Alterando para True para carregar o modelo treinado
    should_render = True  # Alterando para True para visualizar
    render_mode = "human"  # Alterando para "human" para visualização em tempo real

    figure_file = f"plots/{env_name}/{total_episodes}.png"

    best_score = np.float32(-np.inf)  # Usando tipo numpy
    score_history = []

    env = gym.make(env_name, render_mode=render_mode)

    agent = SACAgent(
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0],
        max_action=env.action_space.high,
        checkpoint_dir=f"tmp/sac/{env_name}/{total_episodes}",
    )

    if load_checkpoint:
        env.reset()
        agent.load_models()
        env.render()

    for i in range(total_episodes + 1):
        observation, info = env.reset()
        done = False
        score = np.float32(0)
        frames = []

        while not done:
            # Save the frame to generate a video later
            if render_mode == "rgb_array":
                rgb_array = np.array(env.render())
                frames.append(rgb_array)

            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            score += np.float32(reward)
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

                if render_mode == "rgb_array":
                    # Generate a video of the episode
                    video_path = f"videos/{env_name}/best_episode.gif"
                    create_folder_if_not_exists(video_path)
                    frames_to_gif(frames, video_path, metadata={"Episode": i, "Score": score, "Avg Score": avg_score})

                # Atualiza a curva de aprendizado junto com o GIF
                figure_file = f"plots/{env_name}/learning_curve.png"
                create_folder_if_not_exists(figure_file)
                x = [i + 1 for i in range(i + 1)]  # Atualiza apenas até o episódio atual
                plot_learning_curve(x, score_history[: i + 1], figure_file)

        print(f"[Episode {i}] Score: {score:.2f} Avg Score: {avg_score:.2f}")

        if should_render:
            env.render()

    if not load_checkpoint:
        create_folder_if_not_exists(figure_file)
        x = [i + 1 for i in range(total_episodes)]
        plot_learning_curve(x, score_history, figure_file)

    env.close()
