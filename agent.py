import itertools
import os
import random
from datetime import datetime, timedelta

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn

import flappy_bird_gymnasium
from dqn import DQN
from experience_replay import ReplayMemory
from helpers import preprocess_frame

DATE_FORMAT = "%m-%d %H:%M:%S"
RESULTS_DIR = "training_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

matplotlib.use("Agg")


class DeepQLearningAgent:
    def __init__(self, config_name):
        with open("hyperparameters.yml", "r") as file:
            config = yaml.safe_load(file)
        self.config = config[config_name]

        # if epsilon action is chosen then choose flap with a 1/4 chance
        self.special_action = self.config.get("special_action", False)

        # extract parameters from hyperparameters.yml
        self.environment_id = self.config.get("env_id", "FlappyBird-v0")
        self.learning_rate = self.config.get("learning_rate_a", 0.00025)
        self.discount_factor = self.config.get("discount_factor_g", 0.99)
        self.target_update_interval = self.config.get("network_sync_rate", 1_000)
        self.replay_buffer_capacity = self.config.get("replay_memory_size", 1_000_000)
        self.mini_batch_size = self.config.get("mini_batch_size", 64)
        self.epsilon_start = self.config.get("epsilon_init", 1.0)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.9997)
        self.epsilon_min = self.config.get("epsilon_min", 0.0001)
        self.stop_on_reward = self.config.get("stop_on_reward", 50000)
        self.fc1_nodes = self.config.get("fc1_nodes", 512)
        self.env_make_params = self.config.get("env_make_params", {})
        self.enable_double_dqn = self.config.get("enable_double_dqn", True)
        self.enable_dueling_dqn = self.config.get("enable_dueling_dqn", True)
        self.device = self.config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # set up loss function and optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # set up file paths
        self.LOG_FILE = os.path.join(RESULTS_DIR, f"{config_name}.log")
        self.MODEL_FILE = os.path.join(RESULTS_DIR, f"{config_name}.pt")
        self.GRAPH_FILE = os.path.join(RESULTS_DIR, f"{config_name}.png")

    def play(self):
        env = gym.make(self.environment_id, render_mode="human", **self.env_make_params)

        policy_dqn = DQN(self.fc1_nodes, self.enable_dueling_dqn).to(self.device)

        if self.device == "cpu":
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=torch.device("cpu"))
            )
        else:
            policy_dqn.load_state_dict(
                torch.load(self.MODEL_FILE, map_location=torch.device("cuda"))
            )
        policy_dqn.eval()

        for _ in itertools.count():
            state, _ = env.reset()
            state = preprocess_frame(state)
            state = state.reshape(-1)
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = env.step(action.item())

                new_state = preprocess_frame(new_state)
                new_state = new_state.reshape(-1)
                new_state = torch.tensor(
                    new_state, dtype=torch.float, device=self.device
                )
                state = new_state

                reward = torch.tensor(reward, dtype=torch.float, device=self.device)
                episode_reward += reward

            print(f"The game finished with the score: {episode_reward}")

    def save_training_plot(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))

        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99) : (x + 1)])

        plt.subplot(121)
        plt.ylabel("Mean Rewards")
        plt.plot(mean_rewards)
        plt.subplot(122)
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history)
        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = (
                    rewards
                    + (1 - terminations)
                    * self.discount_factor
                    * target_dqn(new_states)
                    .gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1))
                    .squeeze()
                )
            else:
                target_q = (
                    rewards
                    + (1 - terminations)
                    * self.discount_factor
                    * target_dqn(new_states).max(dim=1)[0]
                )

        current_q = (
            policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        )

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        env = gym.make(self.environment_id, render_mode="human", **self.env_make_params)

        rewards_per_episode = []

        start_time = datetime.now()
        last_graph_update_time = start_time
        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, "w") as file:
            file.write(log_message + "\n")

        policy_dqn = DQN(self.fc1_nodes, self.enable_dueling_dqn).to(self.device)

        epsilon = self.epsilon_start

        memory = ReplayMemory(self.replay_buffer_capacity)

        target_dqn = DQN(self.fc1_nodes, self.enable_dueling_dqn).to(self.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(
            policy_dqn.parameters(), lr=self.learning_rate
        )

        epsilon_history = []

        step_count = 0

        best_reward = float("-inf")

        for episode in itertools.count():
            state, _ = env.reset()
            state = preprocess_frame(state)
            state = state.reshape(-1)
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if random.random() < epsilon:
                    if self.special_action:
                        # if epsilon action is chosen then choose flap with a 1/4 chance
                        action = torch.tensor(
                            random.choices([0, 1], weights=[3, 1], k=1)[0],
                            dtype=torch.int64,
                            device=self.device,
                        )
                    else:
                        action = env.action_space.sample()
                        action = torch.tensor(
                            action, dtype=torch.int64, device=self.device
                        )

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = env.step(action.item())

                new_state = preprocess_frame(new_state)
                new_state = new_state.reshape(-1)
                new_state = torch.tensor(
                    new_state, dtype=torch.float, device=self.device
                )
                state = new_state

                reward = torch.tensor(reward, dtype=torch.float, device=self.device)
                episode_reward += reward

                memory.append((state, action, new_state, reward, terminated))
                step_count += 1

            rewards_per_episode.append(episode_reward)

            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, "a") as file:
                    file.write(log_message + "\n")

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.save_training_plot(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                if step_count > self.target_update_interval:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
