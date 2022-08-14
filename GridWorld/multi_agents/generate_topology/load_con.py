import json

wsize = 5 # number of agents
topo = 1 # fully con

with open('connectivity/%s_%s.json'%(wsize,topo), 'r') as f:
    cdict = json.load(f) # connectivity dict
'''
Sample cdict content for 3 agents and fully-con:

connectivity = 1 means there is connection between these agents

pi = adjacency weight matrix

{
    "graph_type": "FC",
    "experiment id": 1,
    "num_agents": 3,
    "connectivity": [
        [
            1.0,
            1.0,
            1.0
        ],
        [
            1.0,
            1.0,
            1.0
        ],
        [
            1.0,
            1.0,
            1.0
        ]
    ],
    "pi": [
        [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333
        ],
        [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333
        ],
        [
            0.3333333333333333,
            0.3333333333333333,
            0.3333333333333333
        ]
    ]
}

'''