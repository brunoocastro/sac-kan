import gymnasium as gym

from sac import SACAgent

# Configurações
env_name = "InvertedPendulum-v5"
episodes = 5  # Número de episódios para avaliação
render_mode = "human"

# Criar ambiente
env = gym.make(env_name, render_mode=render_mode)

# Criar agente e carregar modelo treinado
agent = SACAgent(
    input_dims=env.observation_space.shape,
    n_actions=env.action_space.shape[0],
    max_action=env.action_space.high,
    checkpoint_dir="tmp/sac/InvertedPendulum-v5/10000",  # Ajuste o caminho conforme necessário
)
agent.load_models()

# Avaliar o agente
for i in range(episodes):
    observation, info = env.reset()
    done = False
    score = 0

    while not done:
        env.render()  # Renderizar o ambiente
        action = agent.choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated

    print(f"Episódio {i+1}: Pontuação = {score}")

env.close()
