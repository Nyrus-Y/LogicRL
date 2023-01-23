import gym

gym.envs.register(
    id='getout',
    entry_point='environments.coinjump.env.CoinJumpEnvV1:CoinJumpEnvV1',
    max_episode_steps=300,
)

gym.envs.register(
    id='getoutplus',
    entry_point='environments.coinjump.env.CoinJumpEnvV2:CoinJumpEnvV2',
    max_episode_steps=300,
)

gym.envs.register(
     id='getoute',
     entry_point='src.environments.coinjump.env.coinJumpEnvE:CoinJumpEnvE',
     max_episode_steps=300,
)

gym.envs.register(
     id='getoutkd',
     entry_point='src.environments.coinjump.env.coinJumpEnvKD:CoinJumpEnvKD',
     max_episode_steps=300,
)