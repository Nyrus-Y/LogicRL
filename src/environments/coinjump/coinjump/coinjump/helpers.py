import random

from .coinjump import CoinJump
from .paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1


def create_coinjump_instance(seed=None, print_seed=False, generator_args=None,**kwargs):
    seed = random.randint(0, 1000000) if seed is None else seed
    if generator_args is None:
        generator_args = {}

    coin_jump = CoinJump(**kwargs)
    level_generator = ParameterizedLevelGenerator_V1(print_seed=print_seed)

    level_generator.generate(coin_jump, seed=seed, **generator_args)
    coin_jump.render()

    return coin_jump
