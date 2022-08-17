import math
import random

from .block import Block
from .coin import Coin
from .coinjump import CoinJump
from .flag import Flag
from .groundEnemy import GroundEnemy
from .powerup import PowerUp


class ParameterizedLevelGenerator:

    def __init__(self, print_seed=True):
        self.print_seed = print_seed

    def generate(self, coinjump: CoinJump, seed=None, generate_enemies=True, dynamic_reward=False,
                 randomize_platform_position=False, spawn_all_entities=False):
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

        if randomize_platform_position:
            platform_start = 5 + rng.randint(0, 10)
        else:
            platform_start = 12

        platform_size = 6

        positions = [
            (4, 2),
            (8, 2),
            (11, 2),
            (platform_start + 2, 5),
            # (platform_start + 4, 5),
            (17, 2)
        ]
        rng.shuffle(positions)
        # 在此取消平台
        for x in range(platform_start, platform_start + platform_size):
            level.add_block(x, 4, snowBlock)

        coinjump.player.x = positions[0][0] - 0.5
        coinjump.player.y = positions[0][1]


        level.entities.append(Coin(level, positions[1][0] - 0.5, positions[1][1], resource_loader=resource_loader))
        level.entities.append(Coin(level, positions[2][0] - 0.5, positions[2][1], resource_loader=resource_loader))
        # level.entities.append(Coin(level, positions[3][0]-0.5, positions[3][1], resource_loader=resource_loader))
        level.entities.append(PowerUp(level, positions[3][0], positions[3][1], resource_loader=resource_loader))

        if generate_enemies:
            level.entities.append(GroundEnemy(level, positions[4][0], positions[4][1], resource_loader=resource_loader))

        if not spawn_all_entities:
            # remove up to 3 entities
            rm_count = 3 - int(math.sqrt(rng.randint(0, 15)))
            for i in range(rm_count):
                # do not remove entity 0, as this is the player
                level.entities.pop(rng.randint(1, len(level.entities) - 1))

        level.entities.append(Flag(level, 24, 2, resource_loader=resource_loader))

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

        level.meta["seed"] = seed
        level.meta["platform_start"] = platform_start
        level.meta["platform_end"] = platform_start + platform_size
        level.meta["reward_coin"] = coinjump.level.reward_values['coin']
        level.meta["reward_powerup"] = coinjump.level.reward_values['powerup']
        level.meta["reward_enemy"] = coinjump.level.reward_values['enemy']
