import random

from .coinjump import CoinJump
from .paramLevelGenerator import ParameterizedLevelGenerator


def create_coinjump_instance(enemy=False, enemies=False,key_door=False, seed=None, print_seed=False, generator_args=None, **kwargs):
    seed = random.randint(0, 1000000) if seed is None else seed
    if generator_args is None:
        generator_args = {}

    coin_jump = CoinJump(**kwargs)
    level_generator = ParameterizedLevelGenerator(enemy=enemy,enemies=enemies,key_door=key_door, print_seed=print_seed)

    level_generator.generate(coin_jump, seed=seed, **generator_args)
    coin_jump.render()

    return coin_jump
