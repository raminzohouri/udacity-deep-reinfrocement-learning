import numpy as np
import time
import sys
from collections import deque
from agent.ddpg_agent import DDPGAgent
from agent_examples import agent_config_default
from utils import (
    log_training_info,
    save_checkpoint,
    get_training_env,
    load_agent_parameters,
)


def ddpg_runner(
    env,
    brain_name="Reacher",
    state_size=33,
    action_size=4,
    n_episodes=100,
    max_t=1000,
    agent_config={},
    target_episodes=100,
    target_score=30,
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
    state_size :
    action_size :
    n_episodes :
    max_t :
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
    scores = []  # episodic scores
    training_info_log = []  # training time and meta data logger
    moving_avg = deque(maxlen=100)  # last 100 scores
    agent_config_default["action_size"] = action_size
    agent_config_default["state_size"] = state_size
    agent = DDPGAgent(**agent_config)
    if run_mode == "test":
        load_agent_parameters(agent, saved_checkpoint_path)
        agent.actor_local.eval()
        agent.critic_local.eval()
    ## Perform n_episodes of training
    training_start_time = time.time()
    for i_episode in range(1, n_episodes + 1):
        episode_start_time = time.time()
        agent.noise.reset()
        states = env.reset(train_mode=run_mode == "train")[
            brain_name
        ].vector_observations
        scores_episode = np.zeros(num_agents)  # rewards per episode for each agent

        for t in range(1, max_t + 1):
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
                for i, (state, action, reward, next_state, done) in enumerate(
                    zip(states, actions, rewards, next_states, dones)
                ):
                    agent.step(state, action, reward, next_state, done)
            # Update the variables for the next iteration
            states = next_states
            scores_episode += rewards

        if run_mode == "test":
            continue
        # Store the rewards and calculate the moving average
        scores.append(scores_episode.tolist())
        moving_avg.append(np.mean(scores[-target_episodes:], axis=0))

        training_info_log.append(
            log_training_info(
                i_episode,
                training_start_time,
                episode_start_time,
                scores_episode,
                moving_avg,
                print_log=print_log,
            )
        )
        ## Check if the environment has been solved
        if moving_avg[-1].mean() >= target_score and i_episode >= target_episodes:
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
            print("\n done.")
            break

    return scores, moving_avg, agent, training_info_log


if __name__ == "__main__":
    if not len(sys.argv) >= 6:
        print(
            "please path the path to environment file, n_episodes, max_t, print_log !"
        )
        exit(0)

    env_file_path = sys.argv[1]
    n_episodes = int(sys.argv[2])
    max_t = int(sys.argv[3])
    print_log = bool(int(sys.argv[4]))
    run_mode = sys.argv[5]
    saved_checkpoint_path = ""
    if run_mode == "test":
        saved_checkpoint_path = sys.argv[6]
    env, num_agents, brain_name, state_size, action_size = get_training_env(
        env_file_path, test_mode=run_mode == "test"
    )

    scores, moving_avg, agent, training_info_log = ddpg_runner(
        env=env,
        brain_name=brain_name,
        state_size=state_size,
        action_size=action_size,
        n_episodes=n_episodes,
        max_t=10,
        agent_config=agent_config_default,
        num_agents=num_agents,
        print_log=print_log,
        run_mode=run_mode,
        saved_checkpoint_path=saved_checkpoint_path,
    )
