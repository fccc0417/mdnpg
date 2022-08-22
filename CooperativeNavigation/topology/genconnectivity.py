"""
Reference:
https://github.com/xylee95/MD-PGT/blob/discrete/generate_topology/genconnectivity.py
"""

import random
import json
import math
random.seed(123)
import numpy as np
import os


def FullyConnectedTopology(num_agents):
    connectivity = []
    for j in range(num_agents):
        neighbors = [1.0 for _ in range(num_agents)]
        connectivity.append(neighbors)
    pi = np.array(connectivity) / num_agents
    return connectivity, pi


def RingTopology(num_agents):
    connectivity = []
    for j in range(num_agents):
        neighbors = [0.0 for _ in range(num_agents)]
        neighbors[j] = 1.0
        neighbors[j - 1] = 1.0
        if j is num_agents - 1:
            neighbors[0] = 1.0
        else:
            neighbors[j + 1] = 1.0
        connectivity.append(neighbors)
    pi = np.array(connectivity) / 3  # Since only 3 agents are active at every situation
    return connectivity, pi


def BiparTopology(num_agents):
    factor = 10 ** 2
    a = math.floor((1 / (num_agents - 1)) * factor) / factor  # round DOWN the 1/(num_agents-1) to 2 decimals
    b = 1 - (num_agents - 2) * a
    c = 1 - 2 * a

    connectivity = []
    pi = []
    # First (num_agents-2) rows:
    for j in range(num_agents - 2):
        neighbors = [0.0 for _ in range(num_agents)]
        neighbors[j] = 1.0
        neighbors[-1] = 1.0
        neighbors[-2] = 1.0
        connectivity.append(neighbors)

        pi_ = [0.0 for _ in range(num_agents)]
        pi_[j] = c
        pi_[-1] = a
        pi_[-2] = a
        pi.append(pi_)

    # Second last row
    neighbors = [1.0 for _ in range(num_agents)]
    neighbors[-1] = 0.0
    connectivity.append(neighbors)

    pi_ = [a for _ in range(num_agents)]
    pi_[-1] = 0.0
    pi_[-2] = b
    pi.append(pi_)

    # Last row
    neighbors = [1.0 for _ in range(num_agents)]
    neighbors[-2] = 0.0
    connectivity.append(neighbors)

    pi_ = [a for _ in range(num_agents)]
    pi_[-1] = b
    pi_[-2] = 0.0
    pi.append(pi_)

    return connectivity, np.array(pi)


def main():
    '''
    New connectivity folder contains json files named with: {num_agents}_{graph_type}.json where
    graph_type 1 = FC
    graph_type 2 = Ring
    graph_type 3 = Bipar (Not implemented yet)
    '''

    connectivity_folder = 'connectivity'
    if not os.path.exists(connectivity_folder):
        os.makedirs(connectivity_folder)

    Topologies = [FullyConnectedTopology, RingTopology, BiparTopology]  # Add BiparTopology when defined
    Topologies_name = ['FC', 'Ring', 'Bipar']  # Add BiparTopology when defined
    agent_num_list = [i for i in range(2, 41, 1)]
    agent_num_list.append(5)
    for num_agents in agent_num_list:
        for eid, topology in enumerate(Topologies):
            connectivity, pi = topology(num_agents)
            cdict = {}
            cdict['graph_type'] = Topologies_name[eid]
            cdict['experiment id'] = eid + 1
            cdict['num_agents'] = int(num_agents)
            cdict['connectivity'] = connectivity
            cdict['pi'] = pi.tolist()
            with open('%s/%s_%s.json' % (connectivity_folder, num_agents, eid + 1), 'w') as f:
                json.dump(cdict, f, sort_keys=False, indent=4)


if __name__ == '__main__':
    main()