# Momentun-based Decentralized Natural Policy Gradient (MDNPG)

Code for paper "[Decentralized Natural Policy Gradient with Variance Reduction for Collaborative Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/2209.02179.pdf)"

## Basic Requirements
* Python (3.7)
* Pytorch (1.4.0)
* Numpy (1.21.5)
* OpenAI Gym (0.18.0)
  

## Code Structure

├─ CooperativeNavigation　# Experiments on Cooperative navigation.  
│　　├─ multiagent　# Code for environment [[REF](https://github.com/openai/multiagent-particle-envs)].       
│　　│　　├─ core.py  
│　　│　　├─ environment.py  
│　　│　　├─ multi_discrete.py  
│　　│　　├─ policy.py  
│　　│　　├─ rendering.py  
│　　│　　├─ scenario.py  
│　　│　　└─ scenarios   
│　　├─ particle_envs　# Code for environment [[REF](https://github.com/openai/multiagent-particle-envs)].      
│　　│　　└─ make_particleworld.py  
│　　├─ topology　# Generate connectivity topology [[REF](https://github.com/xylee95/MD-PGT)].    
│　　│　　├─ connectivity  
│　　│　　├─ genconnectivity.py  
│　　│　　└─ load_con.py  
│　　├─ tools  
│　　│　　└─ tool.py  
│　　├─ nets.py　# Policy network and value network.  
│　　├─ rl_utils.py  
│　　├─ momentum_npg.py　# Class of MDNPG method.  
│　　├─ mdnpg_main.py　# Main file of MDNPG method.  
│　　├─ momentum_pg.py　# Class of MDPGT method [[REF](https://github.com/xylee95/MD-PGT)].  
│　　├─ mdpgt_main.py　# Main file of MDPGT method [[REF](https://github.com/xylee95/MD-PGT)].  
│　　├─ rl_utils_vp.py  
│　　├─ value_propagation_one_step.py　# Class of one-step-version VP method.   
│　　├─ value_propagation_multi_step.py　# Class of multi-step-version VP method.  
│　　└─ value_propagation_main_on_policy.py　# Main file of on-policy-version VP method.    
├─ GridWorld　# Experiments on GridWorld.  
│　　├─ envs　# Code for environment.  
│　　│　　├─ grid_maps  
│　　│　　├─ gridworld.py  
│　　│　　└─ gridworld_4_test.py  
│　　├─ multi_agents　# Multi-agent GridWorld.  
│　　│　　├─ topology　# Generate connectivity topology [[REF](https://github.com/xylee95/MD-PGT)].      
│　　│　　├─ nets.py　# Policy network and value network.    
│　　│　　├─ rl_utils.py  
│　　│　　├─ momentum_npg.py　# Class of MDNPG method.    
│　　│　　├─ mdnpg_main.py　# Main file of MDNPG method.    
│　　│　　├─ momentum_pg.py　# Class of MDPGT method [[REF](https://github.com/xylee95/MD-PGT)].    
│　　│　　├─ mdpgt_main.py　# Main file of MDPGT method [[REF](https://github.com/xylee95/MD-PGT)].    
│　　│　　├─ pg_entropy.py　# Class of PG with entropy method.    
│　　│　　└─ pg_entropy_main.py　# Main file of PG with entropy method.  
│　　├─ single_agent　# Single-agent GridWorld.  
│　　│　　├─ momentum_npg_discrete.py　# Momentum-based NPG.  
│　　│　　├─ momentum_pg_discrete.py　# Momentum-based PG.  
│　　│　　├─ pg_entropy_discrete.py　# PG with entropy.  
│　　│　　├─ ppo_discrete.py　#  PPO.  
│　　│　　└─ srvrnpg_discrete.py　# SRVR-NPG.  
│　　└─ tools  
│　　 　　└─ tool.py  
├─ MountainCar　# Experiments on MountainCarContinuous (single agent).     
│　　├─ momentum_npg_continuous.py　# Momentum-based NPG.   
│　　├─ momentum_pg_continuous.py　# Momentum-based PG.   
│　　├─ ppo_continuous.py　# PPO.  
│　　└─ srvrnpg_continuous.py　# SRVR-NPG.  
└─ README.md  


