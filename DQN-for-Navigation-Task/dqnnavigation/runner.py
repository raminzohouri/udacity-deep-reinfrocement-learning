from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
from agent.dqn_agent import Agent
from agent_examples import *
import sys
from matplotlib import pyplot as plt


def plot_scores(scores, title="Scores"):
    """

    Parameters
    ----------
    scores :
    title :

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = range(0, len(scores), 100)
    y_mean = np.asarray(
        [np.mean(scores[i : i + 100]) for i in range(0, len(scores), 100)]
    )
    y_std = np.asarray(
        [np.std(scores[i : i + 100]) for i in range(0, len(scores), 100)]
    )
    ax.plot(
        x,
        y_mean,
        color="r",
    )
    ax.fill_between(x, y_mean - y_std, y_mean + y_std)
    ax.axhline(y=13, color="k", linestyle="-")
    ax.set_title(title)
    ax.set_ylabel("Mean & STD scores")
    ax.set_xlabel("episode number")
    ax.set_xlim([0, 1000])
    ax.set_xticks(range(0, 1000, 50))


def dqn(
    agent_config,
    eps_start=1.0,
    n_episodes=100,
    eps_end=0.05,
    eps_decay=0.995,
    env=None,
    brain_name=None,
):
    """

    Parameters
    ----------
    agent_config : configuration to build DQN Agent
    eps_start : Epsilon greedy start value
    n_episodes : number of episode
    eps_end : minimum number of epsilon greedy
    eps_decay : factor to decay epsilon greedy
    env : banana environment
    brain_name: brain name of the env object for extracting state information

    Returns
    -------

    """

    agent = Agent(**agent_config)
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    BETA_START = 0.5
    frame_idx = 0
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        beta = min(1.0, BETA_START + i_episode * (1.0 - BETA_START) / agent.buffer_size)
        for t in range(1000):
            action = agent.act(state, eps)
            env_info = env.step(action)[
                brain_name
            ]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done, beta)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        # rl = max(0.00001, rl*eps) # decrease epsilon
        print(
            "\rEpisode {} \t Average Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print("\t eps: {}".format(eps))
        if np.mean(scores_window) >= 13.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
            checkpoint = {
                "agent_config": agent_config,
                "model_state_dict": agent.qnetwork_online.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "number_of_episode": i_episode,
                "scores": scores,
            }
            torch.save(
                checkpoint,
                "./model/checkpoint_{}.pth".format(agent_config["agent_name"]),
            )
            break
    return scores


if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print("please path the path to Banana environment file!")
        exit(0)

    env_file_path = sys.argv[1]
    banana_env = UnityEnvironment(file_name=env_file_path)
    banana_brain = banana_env.brain_names[0]
    scores_v_dqn = dqn(
        agent_config_v_dqn,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    scores_v_dqn_w_bn = dqn(
        agent_configv_dqn_w_bn,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    scores_v_dqn_w_bn_cb = dqn(
        agent_config_v_dqn_w_bn_cb,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    scores_v_dqn_w_bn_per = dqn(
        agent_config_v_dqn_w_bn_per,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    scores_dd_dqn_w_bn_per = dqn(
        agent_config_dd_dqn_w_bn_per,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    scores_dd_dqn_w_bn_per_de = dqn(
        agent_config_dd_dqn_w_bn_per_de,
        eps_start=1.0,
        n_episodes=1000,
        env_file_path=banana_env_file_path,
        env=banana_env,
        brain_name=banana_brain,
    )

    plot_scores(
        scores_v_dqn, title="vanilla dqn with FC q-network & Simple Replay Buffer"
    )
    plot_scores(
        scores_v_dqn_w_bn,
        title="vanilla dqn with batch normalized FC q-network & Simple Replay Buffer",
    )
    plot_scores(
        scores_v_dqn_w_bn_cb,
        title="vanilla dqn with batch normalized FC q-network & Combined Replay Buffer",
    )
    plot_scores(
        scores_v_dqn_w_bn_per,
        title="vanilla dqn with batch normalized FC q-network & Prioritized Replay Buffer",
    )
    plot_scores(
        scores_dd_dqn_w_bn_per,
        title="double dqn with batch normalized FC q-network & Prioritized Replay Buffer",
    )
    plot_scores(
        scores_dd_dqn_w_bn_per_de,
        title="double dqn with batch normalized duelling q-network & Prioritized Replay Buffer",
    )
    plt.show()
