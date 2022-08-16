# Decentralized momentum-based natual policy gradient
Code for paper "....."

## Basic Requirements
* Python (3.7)
* Pytorch (1.4.0)
* Numpy (1.21.5)
* OpenAI Gym (0.18.0)
  

## Code Structure

├─ Cooperative navigation　# Experiments on Cooperative navigation.  
│　　├─ multiagent [[REF](https://github.com/openai/multiagent-particle-envs)]   
│　　│　　├─ core.py  
│　　│　　├─ environment.py  
│　　│　　├─ multi_discrete.py  
│　　│　　├─ policy.py  
│　　│　　├─ rendering.py  
│　　│　　├─ scenario.py  
│　　│　　└─ scenarios   
│　　├─ particle_envs [[REF](https://github.com/openai/multiagent-particle-envs)]  
│　　│　　└─ make_particleworld.py  
│　　├─ topology　# Connectivity topology [[REF](https://github.com/xylee95/MD-PGT)]  
│　　│　　├─ connectivity  
│　　│　　├─ genconnectivity.py  
│　　│　　└─ load_con.py  
│　　├─ tools  
│　　│　　└─ tool.py  
│　　├─ nets.py  
│　　├─ rl_utils.py  
│　　├─ momentum_npg.py  
│　　├─ mdnpg_main.py  
│　　├─ momentum_pg.py  
│　　├─ mdpgt_main.py  
│　　├─ rl_utils_vp.py  
│　　├─ value_propagation_one_step.py  
│　　├─ value_propagation_multi_step.py  
│　　├─ value_propagation_main_on_policy.py  
│　　└─ value_propagation_main_off_policy.py  
├─ GridWorld  
│　　├─ envs  
│　　│　　├─ grid_maps  
│　　│　　├─ gridworld.py  
│　　│　　├─ gridworld_4_test.py  
│　　│　　├─ init_agent_pos_4_all_envs.py  
│　　│　　└─ init_agent_pos_4_single.py  
│　　├─ multi_agents  
│　　│　　├─ generate_topology  
│　　│　　├─ nets.py  
│　　│　　├─ rl_utils.py  
│　　│　　├─ momentum_npg.py  
│　　│　　├─ mdnpg_main.py  
│　　│　　├─ momentum_pg.py  
│　　│　　├─ mdpgt_main.py  
│　　│　　├─ pg_entropy.py  
│　　│　　├─ pg_entropy_main.py  
│　　│　　└─ model_load.py  
│　　├─ single_agent  
│　　│　　├─ model_load.py  
│　　│　　├─ momentum_npg_discrete.py  
│　　│　　├─ momentum_pg_discrete.py  
│　　│　　├─ pg_entropy_discrete.py  
│　　│　　├─ ppo_discrete.py  
│　　│　　└─ svrnpg_discrete.py  
│　　├─ tools  
│　　│　　└─ tool.py  
├─ MountainCar  
│　　├─ momentum_npg_continuous.py  
│　　├─ momentum_pg_continuous.py  
│　　├─ ppo_continuous.py  
│　　├─ rl_utils.py  
│　　└─ svrnpg_continuous.py  
└─ README.md


