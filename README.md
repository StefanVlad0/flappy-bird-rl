# Deep Q-Learning Agent for Flappy Bird

This project was created by [Alexandru Frunză](https://github.com/alexfrunza) and [Vlad Ștefan](https://github.com/StefanVlad0).

![Flappy Bird Agent in Action](imgs/playing.gif)

<br>

### Project Overview

This project implements a **Deep Q-Learning (DQN) agent** to play the **Flappy Bird** game. The agent uses **screenshots of the game as input** and processes them through the CNN (Convolutional Neural Network) to predict the optimal actions.


Key features of the project:
- **Convolutional Neural Network (CNN)** for processing game screenshots.
- **Double DQN support** for improved training stability.
- **Replay Buffer** for experience replay.
- **Epsilon-Greedy Policy** for exploration and exploitation.
- **Configurable Hyperparameters** via a YAML file.
- **Training Progress Visualization** with saved plots.
