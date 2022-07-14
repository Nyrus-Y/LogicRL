
class Block:

    def __init__(self, has_support=True, is_floatable=True, is_passable=False, sprite=None):
        self.has_support = has_support
        self.is_floatable = is_floatable
        self.is_passable = is_passable

        self.sprite = sprite

    def render(self, x, y, camera, frame):
        if self.sprite is None:
            camera.paint_rect(x, y, (1, 1), color=(0, 200, 30))
        else:
            camera.paint_sprite(x, y, (1, 1), self.sprite)


class NoneBlock(Block):

    def __init__(self):
        super().__init__(False, False, True)
