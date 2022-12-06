import gym

gym.envs.register(
    id='CoinJumpEnv-v0',
    entry_point='environments.coinjump.coinjump_learn.env.coinJumpEnv:CoinJumpEnv',
    max_episode_steps=300,
)


gym.envs.register(
    id='CoinJumpEnv-v1',
    entry_point='environments.coinjump.coinjump_learn.env.CoinJumpEnvV1:CoinJumpEnvV1',
    max_episode_steps=300,
)
