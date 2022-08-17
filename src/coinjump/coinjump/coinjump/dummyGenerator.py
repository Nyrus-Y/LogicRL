from .block import Block
from .coin import Coin
from .coinjump import CoinJump
from .flag import Flag
from .groundEnemy import GroundEnemy
from .powerup import PowerUp


class DummyGenerator:

    def __init__(self):
        pass

    def generate(self, coinjump: CoinJump, generate_enemies=True):
        level = coinjump.level
        resource_loader = coinjump.resource_loader

        grassSprite = resource_loader.get_sprite('rock', 'Ground/Grass/grass.png') if resource_loader is not None else None
        snowSprite = resource_loader.get_sprite('snow', 'Ground/Snow/snow.png') if resource_loader is not None else None
        solidBlock = Block(True, False, False, grassSprite)
        snowBlock = Block(True, False, False, snowSprite)
        for x in range(level.width):
            level.add_block(x, 0, solidBlock)
            level.add_block(x, 1, solidBlock)
            level.add_block(x, level.height-1, solidBlock)
            level.add_block(x, level.height-2, solidBlock)

        for y in range(level.height):
            level.add_block(0, y, solidBlock)
            level.add_block(1, y, solidBlock)
            level.add_block(level.width-1, y, solidBlock)
            level.add_block(level.width-2, y, solidBlock)

        for x in range(5,15):
            level.add_block(x, 4, snowBlock)
            if x == 7 or x == 8 or x == 12:
                level.entities.append(Coin(level, x-0.5, 5, resource_loader=resource_loader))

        level.entities.append(PowerUp(level, 17, 2, resource_loader=resource_loader))
        level.entities.append(Flag(level, 24, 2, resource_loader=resource_loader))

        #if generate_enemies:
        #    level.entities.append(GroundEnemy(level, 10, 2, resource_loader=resource_loader))

        coinjump.player.x = 3
        coinjump.player.y = 2
