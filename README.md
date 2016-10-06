dqn-agent
=========

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
* Install requirements with `pip install -r requirements.txt`

Training the AtariAgent
-----------------------
* Train the model with `python AtariAgent.py`

Evaluating the AtariAgent
-------------------------
* Coming soon

Description
-----------
dqn-agent is a [TensorFlow](https://www.tensorflow.org/) implementation of a reinforcement learning agent using [Deep Q Network (DQN)](https://arxiv.org/abs/1312.5602) for [OpenAI Gym](https://gym.openai.com/) environments. The agent learns to play Atari games in the OpenAI Gym Atari environment by repeatedly playing the games and learning to approximate which action (a press of a controller button) gives the most reward in the future. The future reward is approximated using a DQN.

DQN is a multi-layered neural network mapping the game states to expected future rewards. The inputs to the network are screenshots from the game environment. The Atari games have sprites moving across the screen at high speeds so a single screenshot from the environment does not necessarily represent the state of the game accurately (a screenshot of a rolling ball looks the same regardless of the rolling direction). Therefore the network is fed a number of consecutive frames at every timestep to help it distinguish between different movement directions. The output of the network is a vector of values approximating the expected reward of each available action.

Training the agent is computationally demanding. The experiments in the paper are run for 10M or 50M frames. The dqn-agent is currently capable of processing ~30k frames per hour on a GeForce 650m, which means reproducing the smaller experiments (10M frames) takes roughly 2 weeks.

Known differences to the DQN paper
----------------------------------
* The OpenAI Gym Atari environment samples a frame from the game every k frames where k is uniformly sampled from {2,3,4}. In the DQN paper k is fixed.
* In the DQN paper the input pixels are maximum of pixels in two consecutive frames to compensate for sprites only visible every other frame. In dqn-agent the input frames are used as is without applying max to them.

Todos, missing features
-----------------------
* Loading and evaluating pretrained networks.
* OpenAI Gym writeup support and demonstrations of fully trained networks should be added.
* Some naming of parameters and other slight style issues.
* More thorough testing
