# Decentralized momentum-based natual policy gradient
Code for paper "....."

## Basic Requirements
* Python (3.7)
* Pytorch (1.4.0)
* Numpy (1.21.5)
* OpenAI Gym (0.18.0)
  

## Code Structure

├─ Cooperative navigation  # Experiments on Cooperative navigation. [[REF](https://github.com/openai/multiagent-particle-envs)]    
│    ├─ generate_topology  # Connectivity topology [[REF](https://github.com/xylee95/MD-PGT)]  
│    │    ├─ connectivity  
│    │    ├─ genconnectivity.py  
│    │    └─ load_con.py
│    ├─ mNPG_gt_main.py
│    ├─ mPG_gt_main.py
│    ├─ momentum_npg.py
│    ├─ momentum_pg.py
│    ├─ multiagent
│    │    ├─ core.py
│    │    ├─ environment.py
│    │    ├─ multi_discrete.py
│    │    ├─ policy.py
│    │    ├─ rendering.py
│    │    ├─ scenario.py
│    │    └─ scenarios
│    ├─ nets.py
│    ├─ particle_envs
│    │    └─ make_particleworld.py
│    ├─ rl_utils.py
│    ├─ rl_utils_vp.py
│    ├─ tools
│    │    └─ tool.py
│    ├─ value_propagation_main_off_policy.py
│    ├─ value_propagation_main_on_policy.py
│    ├─ value_propagation_multi_step.py
│    └─ value_propagation_one_step.py
├─ GridWorld
│    ├─ envs
│    │    ├─ grid_maps
│    │    ├─ gridworld.py
│    │    ├─ gridworld_4_test.py
│    │    ├─ init_agent_pos_4_all_envs.py
│    │    └─ init_agent_pos_4_single.py
│    ├─ multi_agents
│    │    ├─ generate_topology
│    │    ├─ mNPG_gt_main.py
│    │    ├─ mPG_entropy_main.py
│    │    ├─ mPG_main.py
│    │    ├─ md_npg_gt_results
│    │    ├─ model_load.py
│    │    ├─ momentum_npg.py
│    │    ├─ momentum_pg.py
│    │    ├─ momentum_pg_en.py
│    │    ├─ nets.py
│    │    └─ rl_utils.py
│    ├─ single_agent
│    │    ├─ agents
│    │    ├─ model_load.py
│    │    ├─ momentum_pg_discrete.py
│    │    ├─ momentum_trpo_discrete.py
│    │    ├─ pg_discrete_entropy.py
│    │    ├─ ppo_discrete.py
│    │    └─ svrnpg_discrete.py
│    └─ tools
│           └─ tool.py
├─ MountainCar
│    ├─ momentum_pg_continuous.py
│    ├─ momentum_pg_continuous2.py
│    ├─ momentum_trpo_continuous.py
│    ├─ ppo_continuous.py
│    ├─ rl_utils.py
│    └─ svrnpg_continuous.py
└─ README.md


