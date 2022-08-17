import gym

gym.envs.register(
     id='CoinJumpEnv-v0',
     entry_point='src.coinjump.coinjump_learn.env.coinJumpEnv:CoinJumpEnv',
     max_episode_steps=300,
)

gym.envs.register(
     id='CoinJumpEnvDodge-v0',
     entry_point='src.coinjump.coinjump_learn.env.coinJumpEnvDodge:CoinJumpEnvDodge',
     max_episode_steps=300,
)

gym.envs.register(
     id='CoinJumpEnvKD-v0',
     entry_point='src.coinjump.coinjump_learn.env.coinJumpEnvKD:CoinJumpEnvKD',
     max_episode_steps=300,
)

gym.envs.register(
     id='CoinJumpEnv-v1',
     entry_point='src.coinjump.coinjump_learn.env.coinJumpEnvV1:CoinJumpEnvV1',
     max_episode_steps=300,
)


gym.envs.register(
    id='CoinJumpEnv-v2',
    entry_point='src.coinjump.coinjump_learn.env.coinJumpEnvV2:CoinJumpEnvV2',
    max_episode_steps=300,
)
