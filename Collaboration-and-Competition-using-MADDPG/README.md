# MADDPG for Multi-Agent Collaboration and Competition Task



This project is done as part of my Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Project's goal

The goal is to train an Multi Agent Deep Policy Gradient based agents to learn to play Tennis.
n this environment, two agents control rackets to bounce a ball over a net. I
f an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
Thus, the goal of each agent is to keep the ball in play.
The environment is  called [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis).
The task is episodic, and in order to solve the environment, the agent must get an average score of 0.5 over 100 consecutive episodes.

![](./data/tennis.gif)



### About Reinforcement Learning

> [Reinforcement learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 
>is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal.
> The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. 
>In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, 
>through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most
> important distinguishing features of reinforcement learning. 

This project implement a Multi-Agent Policy Gradient Based reinforcement learning algorithm called [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments, MADDPG](https://papers.nips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf).
In action-value methods we learn the values of actions and
then selected actions based on their estimated action values 1; their policies would not even exist without
the action-value estimates. In policy gradient instead we learn a parameterized policy that can select actions without 
consulting a value function. A value function may still be used to learn the policy parameter, 
but is not required for action selection. MADDPG is an adaptation of actor-critic methods that considers action policies
of other agents and is able to successfully learn policies that require complex multi-agent coordination. Additionally, we introduce a training regimen utilizing an
ensemble of policies for each agent that leads to more robust multi-agent policies.

### Content 

* [data](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/data) : contains project's artifacts and plotting results
* [ddgppositioning](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/maddpgcollaboration): contains project's modular implementation 
  * [agent](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/maddpgcollaboration/agent): contains implementation of the DDPG and MADDPG  
  * [networks](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/maddpgcollaboration/networks): contains implementation of the vanilla Actor-Critic-Network, vanilla Replay Buffer. 
  * [runner.py](https://github.com/raminzohouri/DDGP-for-ArmRobot-Positioning/blob/main/maddpgcollaboration/runner.py): contains implementation of the main training loop of the DDPG agent for learning the Tennis continuous control task.
  * [agent_examples.py](https://github.com/raminzohouri/DDGP-for-ArmRobot-Positioning/blob/main/maddpgcollaboration/agent_examples.py) contains various examples of implemented agents.
* [model](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/model): contains the trained model for various MADDPG agents.
* [notebooks](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/notebook): contains IPython Notebook for solving Tennis project.
* [report](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/report.md): contains the description of the implementation details of the learning algorithms, chosen hyper-parameters, and result explanation. 



### Dependencies  

Add the following python packages which are required.

```python
pip3 install --user torch
pip3 install --user numpy
pip3 install --user matplotlib
```

You do not required to install the Unity ML-Agents environment. In oder to install the environments and right dependencies follow the instruction in:

* [Udacity Deep Reinforcement Learning  repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
* [Udacity Deep RL Project 3: Collaboration and Competition](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)


###  Environment details

The project environment provided by [Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), 
which is originally derived from Unity ML-Agents's [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#tennis).

Note that this environment is similar to, but not identical to environment on the original Unity ML-Agents.

The goal of each agent is to keep the ball in play.

> The Unity Machine Learning Agents Toolkit [(ML-Agents](https://github.com/Unity-Technologies/ml-agents)) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. We provide implementations (based on PyTorch) of state-of-the-art algorithms to enable game developers and hobbyists to easily train intelligent agents for 2D, 3D and VR/AR games. Researchers can also use the provided simple-to-use Python API to train Agents using reinforcement learning, imitation learning, neuroevolution, or any other methods. These trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. The ML-Agents Toolkit is mutually beneficial for both game developers and AI researchers as it provides a central platform where advances in AI can be evaluated on Unity’s rich environments and then made accessible to the wider research and game developer communities. 

Each interaction with environment provided a tuple of **(State, Action, Reward, Next_State, Done)**.

After initialization, `env = UnityEnvironment(file_name="/path-to-env/Tennis.x86_64")`, the environment object offer following important methods:

* `brain_name=env.brain_name[0]`  which are responsible for deciding the actions of their associated agents.
*  `env_info = env.reset(train_mode=True)[brain_name]` rest environment and return initial information dictionary.
* `action_size = brain.vector_action_space_size` returns number of available actions.
* `state = env_info.vector_observations[0]` return observation of the current state.
* `env_info = env.step(action)[brain_name]` applied a particular action and returns successor environment information dictionary.
* `next_state = env_info.vector_observations[0]` extract successor state resulted from applied action. 
* `reward = env_info.rewards[0]` extract reward collected from applied action. 
* `done = env_info.local_done[0]` extracts boolean condition whether system is in terminal state or not.

Note: if the applied action results in a terminal state the **Next_State** is equal to current **State**.

* **State**: The observation space consists of 24 variables, 8 variables corresponding for the position and velocity of the ball and each Tennis racket.
* **Next_State**:  Vector of Observation in continuous space with size of 24.
* **Reward**:  an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  
* **Done**: boolean value, True when applied action results in terminal state.
* **Action**: Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping, clipped between [-1, 1].

### Usage

* The `ipython notebook` provide a stand alone implementation of the algorithms. Fill free to try them out and add your changes and ideas.
* For running the project on you local machine the following steps have to be followed:
  * clone the repository: `git@github.com:raminzohouri/Collaboration-and-Competition-using-MADDPG.git`
  * To train the model and visualize the collected rewards in the project root run:
       * `python3 maddpgcollaboration/runner.py $path-to-tennis-env-file$ #episode #mat_t 0 train`
       * After training the runner will store the trained models in the [model directory](https://github.com/raminzohouri/Collaboration-and-Competition-using-MADDPG/tree/main/model). 
       * Each model check point contain actor and critic network state dictionary, optimizer state dictionary, and collected scores.

### Contributions 

Feel free to fork this repository and or make pull request to in order to add new features.

### License

content of this repository is Copyright © 2020-2020 Ramin Zohouri. It is free software, and may be redistributed under the terms specified in the [LICENSE] file.
