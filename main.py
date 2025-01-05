import argparse

from agent import DeepQLearningAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test model")
    parser.add_argument("config", help="Name of the configuration to use")
    parser.add_argument("--train", help="Training mode", action="store_true")
    args = parser.parse_args()

    agent = DeepQLearningAgent(config_name=args.config)

    if args.train:
        agent.train()
    else:
        agent.play()
