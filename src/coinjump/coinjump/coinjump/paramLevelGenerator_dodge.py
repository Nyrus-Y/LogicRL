import math
import random

from .block import Block
from .coinjump import CoinJump
from .groundEnemy import GroundEnemy


class ParameterizedLevelGenerator_Dodge:

    def __init__(self, print_seed=True):
        self.print_seed = print_seed

    def generate(self, coinjump: CoinJump, seed=None, generate_enemies=True, dynamic_reward=False,
                 spawn_all_entities=False):

        seed = random.randint(0, 500) if seed is None else seed
        rng = random.Random()
        rng.seed(seed, version=3)
        if self.print_seed:
            print("seed", seed)

        level = coinjump.level
        resource_loader = coinjump.resource_loader

        grassSprite = resource_loader.get_sprite('rock',
                                                 'Ground/Grass/grass.png') if resource_loader is not None else None
        snowSprite = resource_loader.get_sprite('snow', 'Ground/Snow/snow.png') if resource_loader is not None else None
        solidBlock = Block(True, False, False, grassSprite)
        snowBlock = Block(True, False, False, snowSprite)

        for x in range(level.width):
            level.add_block(x, 0, solidBlock)
            level.add_block(x, 1, solidBlock)
            level.add_block(x, level.height - 1, solidBlock)
            level.add_block(x, level.height - 2, solidBlock)

        for y in range(level.height):
            level.add_block(0, y, solidBlock)
            level.add_block(1, y, solidBlock)
            level.add_block(level.width - 1, y, solidBlock)
            level.add_block(level.width - 2, y, solidBlock)

        positions = [
            (4, 2),
            (8, 2),
            (11, 2),
            (17, 2)
        ]

        rng.shuffle(positions)

        coinjump.player.x = positions[0][0] - 0.5
        coinjump.player.y = positions[0][1]

        if generate_enemies:
            level.entities.append(GroundEnemy(level, positions[2][0], positions[2][1], resource_loader=resource_loader))

        # setup rewards
        if dynamic_reward:
            reward_value = 5
            coinjump.level.reward_values['coin'] = min(1, rng.randint(0, 3)) * reward_value
            coinjump.level.reward_values['powerup'] = min(1, rng.randint(0, 3)) * reward_value
            coinjump.level.reward_values['enemy'] = min(1, rng.randint(0, 3)) * reward_value

            # with a quarter probability make a reward negative
            if rng.randint(0, 3) == 0:
                neg_reward = ['coin', 'powerup', 'enemy'][rng.randint(0, 2)]
                coinjump.level.reward_values[neg_reward] = -coinjump.level.reward_values[neg_reward]
        else:
            # coinjump1.level.reward_values['coin'] = 5
            coinjump.level.reward_values['coin'] = 3
            coinjump.level.reward_values['powerup'] = 1
            coinjump.level.reward_values['enemy'] = 9
            coinjump.level.reward_values['key'] = 5
            coinjump.level.reward_values['step_depletion'] = 0.1

        level.meta["seed"] = seed
        level.meta["reward_coin"] = coinjump.level.reward_values['coin']
        level.meta["reward_powerup"] = coinjump.level.reward_values['powerup']
        level.meta["reward_enemy"] = coinjump.level.reward_values['enemy']
        level.meta["reward_key"] = coinjump.level.reward_values['key']
