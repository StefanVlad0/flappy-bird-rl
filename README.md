# Deep Q-Learning Agent for Flappy Bird

This project was created by [Alexandru Frunză](https://github.com/alexfrunza) and [Vlad Ștefan](https://github.com/StefanVlad0).

<br>

![Flappy Bird Agent in Action](imgs/playing.gif)

<br>

## Project Overview

This project implements a **Deep Q-Learning (DQN) agent** to play the **Flappy Bird** game. The agent uses **screenshots of the game as input** and processes them through the CNN (Convolutional Neural Network) to predict the optimal actions.


Key features of the project:
- **Convolutional Neural Network (CNN)** for processing game screenshots.
- **Double DQN support** for improved training stability.
- **Replay Buffer** for experience replay.
- **Epsilon-Greedy Policy** for exploration and exploitation.
- **Configurable Hyperparameters** via a YAML file.
- **Training Progress Visualization** with saved plots.

<br>

## Installation Tutorial

### Step 1: Clone the Repository

```bash
git clone https://github.com/StefanVlad0/flappy-bird-rl.git
cd flappy-bird-rl
```

### Step 2: Create a Virtual Environment and Install Dependencies
Create a virtual environment to isolate your project dependencies:
```bash
python -m venv venv
```
Activate the virtual environment:
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Step 3: Train/Test the Agent
Run the following command to start training the agent:
```bash
python main.py --train alex6
```

This will start the training process using the **`alex6`** hyperparameter set defined in the **`hyperparameters.yml`** file.

Or test the agent:
```bash
python main.py alex6
```

## Input Preprocessing

The **screenshots of the game** are preprocessed before being fed into the neural network.

First, masks are applied to **remove irrelevant background elements** (like sky, grass, and clouds), leaving only key objects such as the bird and pipes.

Then, the frame is converted to **grayscale** to simplify the representation and reduce unnecessary information.

<img src="imgs/frame_grayscaled_without_bg.png" width="270" height="480">

Additionally, a crop is applied to **remove the ground area**, as it contains redundant information.
Lastly, the processed frame is resized to **64x64 pixels** and normalized to values between 0 and 1.
   
<img src="imgs/scaled_frame.png" width="256" height="256">