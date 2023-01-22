import gym

gym.envs.register(
    id='CoinJumpEnv-v1',
    entry_point='environments.coinjump.env.CoinJumpEnvV1:CoinJumpEnvV1',
    max_episode_steps=300,
)

gym.envs.register(
    id='CoinJumpEnv-v2',
    entry_point='environments.coinjump.env.CoinJumpEnvV2:CoinJumpEnvV2',
    max_episode_steps=300,
)

gym.envs.register(
     id='CoinJumpEnvE-v0',
     entry_point='src.environments.coinjump.env.coinJumpEnvE:CoinJumpEnvE',
     max_episode_steps=300,
)

gym.envs.register(
     id='CoinJumpEnvKD-v0',
     entry_point='src.environments.coinjump.env.coinJumpEnvKD:CoinJumpEnvKD',
     max_episode_steps=300,
)