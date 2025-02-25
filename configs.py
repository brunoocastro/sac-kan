from trainer_handler import TrainingConfig


def get_configs():
    return [
        TrainingConfig(
            environment="InvertedPendulum-v5",
            episodes=100,
            render_mode="rgb_array",
            load_checkpoint=False,
            seeds=[1],
            debug_mode=False,
        ),
        TrainingConfig(
            environment="InvertedPendulum-v5",
            episodes=1000,
            render_mode="rgb_array",
            load_checkpoint=False,
            seeds=[1],
            debug_mode=False,
        ),
        TrainingConfig(
            environment="InvertedPendulum-v5",
            episodes=10000,
            render_mode="rgb_array",
            load_checkpoint=False,
            seeds=[1],
            debug_mode=False,
        ),
        # TrainingConfig(
        #     environment="InvertedPendulum-v5",
        #     episodes=100000,
        #     render_mode="rgb_array",
        #     load_checkpoint=False,
        #     seeds=[1],
        #     debug_mode=False,
        # ),
        # TrainingConfig(
        #     environment="InvertedPendulum-v5",
        #     episodes=1e6,
        #     render_mode="rgb_array",
        #     load_checkpoint=False,
        #     seeds=[1],
        #     debug_mode=False,
        # ),
        # TrainingConfig(
        #     environment="InvertedDoublePendulum-v5",
        #     episodes=10000,
        #     render_mode="rgb_array",
        #     load_checkpoint=False,
        #     seeds=[1, 2],
        # ),
    ]
