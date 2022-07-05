import math
import random

from .block import Block
from .coinjump import CoinJump
from .door import Door
from .key import Key
from .groundEnemy import GroundEnemy


class ParameterizedLevelGenerator_V1:

    def __init__(self, print_seed=True):
        self.print_seed = print_seed

    def generate(self, coinjump: CoinJump, seed=None, dynamic_reward=False, spawn_all_entities=False):

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
            (3, 2),
            (7, 2),
            (11, 2),
            (15, 2),
            (19, 2),
            (23, 2)
        ]

        rng.shuffle(positions)

        coinjump.player.x = positions[0][0] - 0.5
        coinjump.player.y = positions[0][1]

        level.entities.append(Key(level, positions[1][0] - 0.5, positions[1][1], resource_loader=resource_loader))
        level.entities.append(GroundEnemy(level, positions[2][0], positions[2][1], resource_loader=resource_loader))
        level.entities.append(GroundEnemy(level, positions[4][0], positions[4][1], resource_loader=resource_loader))
        #level.entities.append(Door(level, 24, 2, resource_loader=resource_loader))
        level.entities.append(Door(level,positions[3][0], positions[3][1], resource_loader=resource_loader))

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
            # coinjump.level.reward_values['coin'] = 5
            coinjump.level.reward_values['coin'] = 3
            coinjump.level.reward_values['powerup'] = 1
            coinjump.level.reward_values['enemy'] = 9
            coinjump.level.reward_values['key'] = 10
            coinjump.level.reward_values['door'] = 20

        level.meta["seed"] = seed
        level.meta["reward_coin"] = coinjump.level.reward_values['coin']
        level.meta["reward_powerup"] = coinjump.level.reward_values['powerup']
        level.meta["reward_enemy"] = coinjump.level.reward_values['enemy']
        level.meta["reward_key"] = coinjump.level.reward_values['key']
        level.meta["reward_door"] = coinjump.level.reward_values['door']
