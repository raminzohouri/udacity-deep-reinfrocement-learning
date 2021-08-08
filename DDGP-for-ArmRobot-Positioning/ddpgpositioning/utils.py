import numpy as np
import time
import torch
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment


def get_training_env(env_file_path, test_mode):
    """

    Parameters
    ----------
    env_file_path :
    test_mode:
    Returns
    -------

    """
    env = UnityEnvironment(
        base_port=5445,
        curriculum=None,
        seed=42,
        docker_training=False,
        no_graphics=not test_mode,
        file_name=env_file_path,
    )
    # prep the env
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=not test_mode)[brain_name]
    num_agents = len(env_info.agents)
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    return env, num_agents, brain_name, state_size, action_size


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


def log_training_info(
    i_episode,
    training_start_time,
    episode_start_time,
    scores_episode,
    moving_avg,
    print_log=False,
):
    """

    Parameters
    ----------
    i_episode :
    training_start_time :
    episode_start_time :
    scores_episode :
    moving_avg :
    print_log :

    Returns
    -------

    """
    # Calculate the elapsed time
    episode_duration = time.time() - episode_start_time
    elapsed_duration = time.time() - training_start_time
    episode_duration_str = time.strftime("%Mm%Ss", time.gmtime(episode_duration))

    if print_log:
        print(
            "Episode {:3d} ({})\tScore: {:5.2f} (H: {:5.2f} / L: {:5.2f})\t"
            "Moving average: {:5.2f} (H: {:5.2f} / L: {:5.2f})".format(
                i_episode,
                episode_duration_str,
                scores_episode.mean(),
                scores_episode.max(),
                scores_episode.min(),
                moving_avg[-1].mean(),
                moving_avg[-1].max(),
                moving_avg[-1].min(),
            )
        )
    return {
        "training_start_time": training_start_time,
        "episode_start_time": episode_start_time,
        "episode_duration": episode_duration,
        "elapsed_duration": elapsed_duration,
        "scores_episode_mean": scores_episode.mean(),
        "scores_episode_max": scores_episode.max(),
        "scores_episode_min": scores_episode.min(),
        "moving_avg_mean": moving_avg[-1].mean(),
        "moving_avg_max,": moving_avg[-1].max(),
        "moving_avg_min": moving_avg[-1].min(),
    }


def save_checkpoint(
    i_episode,
    scores,
    moving_avg,
    training_info_log,
    agent,
    agent_config,
    training_start_time,
    target_episodes,
):
    """

    Parameters
    ----------
    i_episode :
    scores :
    moving_avg :
    training_info_log :
    agent :
    agent_config :
    training_start_time :
    target_episodes :

    Returns
    -------

    """

    elapsed_duration_str = time.strftime(
        "%Hh%Mm%Ss", time.gmtime(time.time() - training_start_time)
    )
    print(
        "\nEnvironment solved in {:d} episodes!\t"
        "Average Score: {:.2f}\tElapsed time: {}".format(
            i_episode - target_episodes,
            moving_avg[-1].mean(),
            elapsed_duration_str,
        )
    )
    checkpoint = {
        "agent_config": agent_config,
        "actor_state_dict": agent.actor_local.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_state_dict": agent.critic_local.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "number_of_episode": i_episode,
        "scores": scores,
        "moving_avg": moving_avg,
        "training_info_log": training_info_log,
    }
    torch.save(
        checkpoint,
        "./model/checkpoint_done_{}_episode_{}.pth".format(
            agent_config["agent_name"], i_episode
        ),
    )


def load_agent_parameters(agent, saved_checkpoint_path):
    """

    Parameters
    ----------
    agent :
    saved_checkpoint_path :

    Returns
    -------

    """
    agent_checkpoint = torch.load(saved_checkpoint_path)
    agent.actor_local.load_state_dict(agent_checkpoint["actor_state_dict"])
    agent.critic_local.load_state_dict(agent_checkpoint["critic_state_dict"])
    agent.actor_optimizer.load_state_dict(
        agent_checkpoint["actor_optimizer_state_dict"]
    )
    agent.critic_optimizer.load_state_dict(
        agent_checkpoint["critic_optimizer_state_dict"]
    )
