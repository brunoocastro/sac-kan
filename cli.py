import argparse

from configs import get_configs
from trainer_handler import TrainerHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the SAC agent")
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the agent")
    parser.add_argument("--render", action="store_true", help="Render the agent")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()


def main():
    args = parse_args()
    configs = get_configs()

    for config in configs:
        if args.render:
            config.render_mode = "human"

        if args.debug:
            config.debug_mode = True

    if args.train:
        trainer = TrainerHandler(configs)
        trainer.run_all_trainings()

    elif args.evaluate:
        for config in configs:
            config.load_checkpoint = True

        trainer = TrainerHandler(configs)
        trainer.run_all_evaluations()


if __name__ == "__main__":
    main()
