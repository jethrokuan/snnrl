from gym.envs.registration import register

register(
    id='ImageCartPole-v0',
    entry_point='snnrl.envs:ImageCartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='ImageCartPole-v1',
    entry_point='snnrl.envs:ImageCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)
