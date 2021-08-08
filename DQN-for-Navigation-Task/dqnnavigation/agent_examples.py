agent_config_v_dqn = {
    "agent_name": "v_dqn",
    "agetn_desription": "vanilla dqn with simple FC q network",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": False,
    "combined_reply_buffer": False,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": False,
    "double_dqn": False,
    "dueling_dqn": False,
}

agent_config_v_dqn_w_bn = {
    "agent_name": "v_dqn_w_bn",
    "agetn_desription": "vanilla dqn with batch normalized FC q-network & Simple Replay "
    "Buffer",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": False,
    "combined_reply_buffer": False,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": True,
    "double_dqn": False,
    "dueling_dqn": False,
}

agent_config_v_dqn_w_bn_cb = {
    "agent_name": "v_dqn_w_bn_cb",
    "agetn_desription": "vanilla dqn with batch normalized FC q-network & Combined Replay "
    "Buffer",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": False,
    "combined_reply_buffer": True,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": True,
    "double_dqn": False,
    "dueling_dqn": False,
}

agent_config_v_dqn_w_bn_per = {
    "agent_name": "v_dqn_w_bn_per",
    "agetn_desription": "vanilla dqn with batch normalized FC q-network & Prioritized "
    "Replay Buffer",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": True,
    "combined_reply_buffer": False,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": True,
    "double_dqn": False,
    "dueling_dqn": False,
}

agent_config_dd_dqn_w_bn_per = {
    "agent_name": "dd_dqn_w_bn_per",
    "agetn_desription": "double dqn with batch normalized FC q-network & Prioritized Replay "
    "Buffer",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": True,
    "combined_reply_buffer": False,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": True,
    "double_dqn": True,
    "dueling_dqn": False,
}

agent_config_dd_dqn_w_bn_per_de = {
    "agent_name": "dd_dqn_w_bn_per_de",
    "agetn_desription": "double dqn with batch normalized duelling q-network & Prioritized "
    "Replay Buffer",
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "use_per": True,
    "combined_reply_buffer": False,
    "update_every": 1,
    "target_network_update_freq": 4,
    "buffer_size": int(1e4),
    "with_batch_norm": True,
    "double_dqn": True,
    "dueling_dqn": True,
}
