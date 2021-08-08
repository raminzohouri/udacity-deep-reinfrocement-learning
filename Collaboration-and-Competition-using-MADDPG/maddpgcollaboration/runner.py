import numpy as np
import time
import sys
from collections import deque
from agent.maddpg import MADDPG
from agent_examples import maddpg_agent_config_default, maddpg_agent_config_per
from utils import (
    log_training_info,
    save_checkpoint,
    get_training_env,
    load_agent_parameters,
)


def maddpg_runner(
    env,
    brain_name="Reacher",
    n_episodes=100,
    agent_config={},
    target_episodes=100,
    target_score=0.5,
    num_agents=20,
    print_log=False,
    run_mode="train",
    saved_checkpoint_path="model/checkpoint.pth",
):
    """
    Deep Deterministic Policy Gradients (DDPG).
    Parameters
    ----------
    env :
    brain_name :
    n_episodes :
    agent_config :
    target_episodes :
    target_score :
    num_agents :
    print_log :
    run_mode:
    saved_checkpoint_path:
    Returns
    -------

    """
    saved_good = False
    scores = []  # episodic scores
    training_info_log = []  # training time and meta data logger
    moving_avg = deque(maxlen=100)  # last 100 scores
    agent = MADDPG(**agent_config)

    if run_mode == "test":
        load_agent_parameters(agent, saved_checkpoint_path)
        for a in agent.agents:
            a.actor_local.eval()
            a.critic_local.eval()
    ## Perform n_episodes of training
    training_start_time = time.time()
    BETA_START = 0.5
    for i_episode in range(1, n_episodes + 1):
        beta = min(1.0, BETA_START + i_episode * (1.0 - BETA_START) / 1e6)
        states = env.reset(train_mode=run_mode == "train")[
            brain_name
        ].vector_observations
        scores_episode = np.zeros(num_agents)  # rewards per episode for each agent
        episode_start_time = time.time()
        agent.reset()
        while True:
            # Perform a step: S;A;R;S'
            actions = agent.act(states)  # select the next action for each agent
            env_info = env.step(actions)[
                brain_name
            ]  # send the actions to the environment
            rewards = env_info.rewards  # get the rewards
            next_states = env_info.vector_observations  # get the next states
            dones = env_info.local_done  # see if episode has finished
            # Send the results to the Agent
            if run_mode == "train":
                agent.step(states, actions, rewards, next_states, dones, beta)
            # Update the variables for the next iteration
            states = next_states
            scores_episode += rewards
            # break if any agents are done
            if np.any(dones):
                break

        if run_mode == "test":
            continue
        # Store the rewards and calculate the moving average
        score = np.max(scores_episode)
        scores.append(score)
        moving_avg.append(score)

        training_info_log.append(
            log_training_info(
                i_episode,
                training_start_time,
                episode_start_time,
                scores,
                moving_avg,
                print_log=print_log,
            )
        )
        ## Check if the environment has been solved
        # if np.mean(moving_avg) >= target_score and i_episode >= target_episodes:
        if i_episode % target_episodes == 0:
            save_checkpoint(
                i_episode,
                scores,
                moving_avg,
                training_info_log,
                agent,
                agent_config,
                training_start_time,
                target_episodes,
            )
        if np.mean(moving_avg) >= target_score and not saved_good:

            save_checkpoint(
                i_episode,
                scores,
                moving_avg,
                training_info_log,
                agent,
                agent_config,
                training_start_time,
                target_episodes,
            )
            saved_good = True
            print("\n checkpoint at episode {} saved.".format(i_episode))
            # break

    return scores, moving_avg, agent, training_info_log


if __name__ == "__main__":
    if not len(sys.argv) >= 5:
        print(
            "please path the path to environment file, n_episodes, max_t, print_log !"
        )
        exit(0)

    env_file_path = sys.argv[1]
    n_episodes = int(sys.argv[2])
    print_log = bool(int(sys.argv[3]))
    run_mode = sys.argv[4]
    saved_checkpoint_path = ""
    if run_mode == "test":
        saved_checkpoint_path = sys.argv[5]
    env, num_agents, brain_name, state_size, action_size = get_training_env(
        env_file_path, test_mode=run_mode == "test"
    )
    maddpg_agent_config_default["action_size"] = action_size
    maddpg_agent_config_default["state_size"] = state_size
    scores, moving_avg, agent, training_info_log = maddpg_runner(
        env=env,
        brain_name=brain_name,
        n_episodes=n_episodes,
        agent_config=maddpg_agent_config_default,
        num_agents=num_agents,
        print_log=print_log,
        run_mode=run_mode,
        saved_checkpoint_path=saved_checkpoint_path,
    )
