# sac-kan

A comparison between the standard implementation of [Soft Actor Critic (using MLP)](https://arxiv.org/abs/1801.01290) and a new one with [KAN (Kolmogorov-Arnold Network)](https://github.com/mintisan/awesome-kan)

## CLI

All commands will run the SAC agent using the configurations defined in [`configs.py`](./configs.py).

You could use the flag `--render` to see the Human Mode rendering (bypassing the configs).

The CLI provides two main options:

### Train the agent

```bash
python cli.py --train
```

### Evaluate the agent

```bash
python cli.py --evaluate
```

This command will load a trained model and evaluate (visually) its performance in the environment.

## Run Locally

To run locally, you need to install the dependencies. Do that running:

```bash
make setup
```

## TODO LIST

- [x] Implement a method that apply the same seed for running experiments
- [ ] Create SAC Agent with KAN at Value networks
- [ ] Define the exact environments where the experiment will be ran at
- [ ] Define the metrics to the paper
