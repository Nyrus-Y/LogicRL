import gym

gym.envs.register(
    id='CoinJumpEnv-v1',
    entry_point='environments.coinjump.env.CoinJumpEnvV1:CoinJumpEnvV1',
    max_episode_steps=300,
)
