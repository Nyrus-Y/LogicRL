import random

from .coinjump import CoinJump
from .paramLevelGenerator import ParameterizedLevelGenerator
from .paramLevelGenerator_dodge import ParameterizedLevelGenerator_Dodge
from .paramLevelGenerator_keydoor import ParameterizedLevelGenerator_KeyDoor
from .paramLevelGenerator_V1 import ParameterizedLevelGenerator_V1


def create_coinjump_instance(seed=None, print_seed=False, generator_args=None, **kwargs):
    seed = random.randint(0, 1000000) if seed is None else seed
    if generator_args is None:
        generator_args = {}

    # coin_jump = CoinJump(Dodge_mode=True, **kwargs)
    coin_jump = CoinJump(V1=True, **kwargs)

    level_generator = ParameterizedLevelGenerator_V1(print_seed=print_seed)
    #level_generator = ParameterizedLevelGenerator_Dodge(print_seed=print_seed)
    # level_generator = ParameterizedLevelGenerator_KeyDoor(print_seed=print_seed)
    #level_generator = ParameterizedLevelGenerator(print_seed=print_seed)

    level_generator.generate(coin_jump, seed=seed, **generator_args)
    coin_jump.render()

    return coin_jump
