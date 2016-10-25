dqn-agent
=========

![Agent playing Breakout](/images/breakout.gif?raw=true)

Requirements
------------
* pip
* virtualenv (recommended)
* python 3
* numpy
* tensorflow
* OpenAI Gym with Atari environments

Installation
------------
* Clone this repository
* Create new virtualenv environment
* For GPU enabled version of tensorflow, check installation instructions on the tensorflow [site](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html)
* For Mac: Install requirements with `pip install -r requirements.txt`
* For other OSs edit requirements.txt with correct tensorflow version

Training the AtariAgent
-----------------------
* Train the model with `python AtariAgent.py`
* You can select the environment with `--env <ENV_NAME>`
* Enable rendering with `--render`

Evaluating the AtariAgent
-------------------------
* Evaluate agent using `python AtariAgent.py --evaluate`
* Load trained weights with `--load_weights <PATH_TO_WEIGHTS>`

Description
-----------
In the animation at the top of the page the Agent is playing breakout after ~10M frames, which corresponds to ~30 hours of training on Geforce GTX 770.

dqn-agent is a [TensorFlow](https://www.tensorflow.org/) implementation of a reinforcement learning agent using [Deep Q Network (DQN)](https://arxiv.org/abs/1312.5602) for [OpenAI Gym](https://gym.openai.com/) environments. The agent learns to play Atari games in the OpenAI Gym Atari environment by repeatedly playing the games and learning to approximate which action (a press of a controller button) gives the most reward in the future. The future reward is approximated using a DQN.

Max Q value                                                 |  Loss
:----------------------------------------------------------:|:-------------------------------------------------:
![Max Q value during training](/images/max_q.png?raw=true)  | ![Loss during training](/images/loss.png?raw=true)

The maximum Q value during training and associated loss are displayed in the graphs above.

Known differences to the DQN paper
----------------------------------
* The OpenAI Gym Atari environment samples a frame from the game every k frames where k is uniformly sampled from {2,3,4}. In the DQN paper k is fixed.
* In the DQN paper the input pixels are maximum of pixels in two consecutive frames to compensate for sprites only visible every other frame. In dqn-agent the input frames are used as is without applying max to them.
* In the Breakout-v0 environment dqn-agent achieves average reward of 75 over 100 episodes, whereas the DQN agent in the paper achieves avg reward of over 165. This could be due to differences in the game environment or due to some implementation difference.
