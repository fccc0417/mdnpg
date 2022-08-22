import numpy as np


def agent_pos_reset_4_envs(envs):
    """Randomly initialize a common initial position for all agents."""
    goal_pos_list = []
    obs_pos_list = []
    for env in envs:
        goal_pos_list.append(env.goal_pos)
        obs_pos_list.append(np.squeeze(env.obstacle_pos))
    goal_poss = np.array(goal_pos_list)
    obs_poss = np.concatenate(obs_pos_list, axis=0)
    all_poss = np.concatenate((goal_poss, obs_poss), axis=0)
    while True:
        agent_pos = np.random.randint(0, 10, envs[0].pos_dim)
        flag = False
        for pos in all_poss:
            if (agent_pos == pos).all():
                flag = True
                break
        if flag:
            continue
        else:
            for env in envs:
                env.init_pos = agent_pos
            break
    return agent_pos