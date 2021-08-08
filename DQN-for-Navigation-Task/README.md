# DQN for Navigation Task



This project is done as part of my Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### Project's goal

The goal is to train an DQN agent that navigates in a virtual environment and collect as many yellow bananas as possible while avoiding blue bananas.
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

![](./data/banana.gif)



### About Deep Reinforcement Learning

> [Reinforcement learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning.

This project implement a Value Based reinforcement learning algorithm called [Deep Q-Networks](https://deepmind.com/research/dqn/) 

### Content 

* [data](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/data) : contains project's artifacts and plotting results
* [dqnnavigation](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/dqnnavigation): contains project's modular implementation 
  * [agent](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/dqnnavigation/agent): contains implementation of the DQN, Double-DQN, 
  * [networks](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/dqnnavigation/networks): contains implementation of the vanilla Q-Network, Dueling-Q-Network, vanilla Replay Buffer, and Prioritized Replay Buffer. 
  * [runner.py](https://github.com/raminzohouri/DQN-for-Navigation-Task/blob/main/dqnnavigation/runner.py): contains implementation of the mail training loop of the DQN agent for Banana Collector navigation task.
  * [agent_examples.py](https://github.com/raminzohouri/DQN-for-Navigation-Task/blob/main/dqnnavigation/agent_examples.py) contains various examples of implemented agents.
* [model](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/model): contains the trained model for various DQN agents.
* [notebooks](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/notebook): contains IPython Notebook of the navigation project.
* [report](https://github.com/raminzohouri/DQN-for-Navigation-Task/blob/main/report.md): contains the description of the implementation details of the learning algorithms, chosen hyper-parameters, and result explanation. 



### Dependencies  

Add the following python packages which are required.

```python
pip3 install --user torch
pip3 install --user numpy
pip3 install --user matplotlib
```

You do not required to install the Unity ML-Agents environment. In oder to install the environments and right dependencies follow the instruction in:

* [Udacity Deep Reinforcement Learning  repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
* [Udacity Deep RL Navigation 1 Project](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started)



###  Environment details

The project environment provided by [Udacity](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), which is originally derived from Unity ML-Agents's [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).

Note that this environment is similar to, but not identical to environment on the original Unity ML-Agents.

> The Unity Machine Learning Agents Toolkit [(ML-Agents](https://github.com/Unity-Technologies/ml-agents)) is an open-source project that enables games and simulations to serve as environments for training intelligent agents. We provide implementations (based on PyTorch) of state-of-the-art algorithms to enable game developers and hobbyists to easily train intelligent agents for 2D, 3D and VR/AR games. Researchers can also use the provided simple-to-use Python API to train Agents using reinforcement learning, imitation learning, neuroevolution, or any other methods. These trained agents can be used for multiple purposes, including controlling NPC behavior (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release. The ML-Agents Toolkit is mutually beneficial for both game developers and AI researchers as it provides a central platform where advances in AI can be evaluated on Unity’s rich environments and then made accessible to the wider research and game developer communities. 

Each interaction with environment provided a tuple of **(State, Action, Reward, Next_State, Done)**.

After initialization, `env = UnityEnvironment(file_name="/path-to-env/Banana.x86_64")`, the environment object offer following important methods:

* `brain_name=env.brain_name[0]`  which are responsible for deciding the actions of their associated agents.
*  `env_info = env.reset(train_mode=True)[brain_name]` rest environment and return initial information dictionary.
* `action_size = brain.vector_action_space_size` returns number of available actions.
* `state = env_info.vector_observations[0]` return observation of the current state.
* `env_info = env.step(action)[brain_name]` applied a particular action and returns successor environment information dictionary.
* `next_state = env_info.vector_observations[0]` extract successor state resulted from applied action. 
* `reward = env_info.rewards[0]` extract reward collected from applied action. 
* `done = env_info.local_done[0]` extracts boolean condition whether system is in terminal state or not.

Note: if the applied action results in a terminal state the **Next_State** is equal to current **State**.

* **State**: Vector of Observation in discrete space with size of 37 and it contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
* **Next_State**:  Vector of Observation in discrete space with size of 37.
* **Reward**:  +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
* **Done**: boolean value, True when applied action results in terminal state.
* **Action**: discrete integer number which enables agent to move in four different direction:
    - 0 - move forward.
    - 1 - move backward.
    - 2 - turn left.
    - 3 - turn right.

### Usage

* The `ipython notebook` provide a stand alone implementation of the algorithms. Fill free to try them out and add your changes and ideas.
* For running the project on you local machine the following steps have to be followed:
  * clone the repository: `git@github.com:raminzohouri/DQN-for-Navigation-Task.git`
  * To train the model and visualize the collected rewards in the project root run:
       * `python3 dqnnavigation/runner.py $path-to-banana-env-file$`
       * After training the runner will store the trained modes in the [mode directory](https://github.com/raminzohouri/DQN-for-Navigation-Task/tree/main/model). 
       * Each model check point contain Q-network state dictionary, optimizer state dictionary, and collected scores.

### Contributions 

Feel free to fork this repository and or make pull requerst to in order to add new features.

### License

content of this repository is Copyright © 2020-2020 Ramin Zohouri. It is free software, and may be redistributed under the terms specified in the [LICENSE] file.
