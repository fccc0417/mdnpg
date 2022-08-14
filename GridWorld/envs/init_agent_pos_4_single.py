import numpy as np


def agent_pos_reset(env):
    while True:
        agent_pos = np.random.randint(0, 10, 2)
        flag = False
        if (agent_pos == env.goal_pos).all():
            continue

        for pos in env.obstacle_pos:
            if (agent_pos == pos).all():
                flag = True
                break
        if flag:
            continue
        else:
            env.init_pos = agent_pos
            break

    return agent_pos