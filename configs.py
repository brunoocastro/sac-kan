from trainer_handler import TrainingConfig


def get_configs():
    return [
        TrainingConfig(
            environment="InvertedPendulum-v5",
            episodes=10000,
            render_mode="rgb_array",
            load_checkpoint=False,
            seeds=[1, 2],
        ),
        TrainingConfig(
            environment="InvertedDoublePendulum-v5",
            episodes=10000,
            render_mode="rgb_array",
            load_checkpoint=False,
            seeds=[1, 2],
        ),
    ]
