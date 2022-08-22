from gym.envs.registration import register

# Multiagent particle_envs
# ----------------------------------------

register(
    id='MultiagentSimple-v0',
    entry_point='multiagent.particle_envs:SimpleEnv',
    # FIXME(cathywu) currently has to be exactly max_path_length parameters in
    # rllab run script
    max_episode_steps=100,
)

register(
    id='MultiagentSimpleSpeakerListener-v0',
    entry_point='multiagent.particle_envs:SimpleSpeakerListenerEnv',
    max_episode_steps=100,
)
