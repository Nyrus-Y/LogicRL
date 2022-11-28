import gym

gym.envs.register(
    id='CoinJumpEnv-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.coinJumpEnv:CoinJumpEnv',
    max_episode_steps=300,
)

gym.envs.register(
    id='CoinJumpEnvDodge-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.coinJumpEnvD:CoinJumpEnvD',
    max_episode_steps=300,
)

gym.envs.register(
    id='CoinJumpEnvKD-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.coinJumpEnvKD:CoinJumpEnvKD',
    max_episode_steps=300,
)

gym.envs.register(
    id='CoinJumpEnvNeural-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.CoinJumpEnvNeural:CoinJumpEnvNeural',
    max_episode_steps=300,
)

gym.envs.register(
    id='CoinJumpEnvLogic-v0',
    entry_point='src.environments.coinjump.coinjump_learn.env.CoinJumpEnvLogic:CoinJumpEnvLogic',
    max_episode_steps=300,
)
