import glfw
import gym3
from procgen import ProcgenGym3Env
# glfw.init()

_key_to_name = {
            getattr(glfw, attr): attr.split("_", 1)[1]
            for attr in dir(glfw)
            if attr.startswith("KEY_")
        }

class InteractiveEnv(gym3.ViewerWrapper):
    def __init__(self, env_name):
        env = ProcgenGym3Env(num=1, env_name=env_name, render_mode="rgb_array")
        super().__init__(env, info_key="rgb")
        glfw.set_key_callback(self._renderer._window, self._on_key_event)
        self._keys_pressed, self._keys_clicked = set(), set()
        self.keys_to_act = lambda keys: env.callmethod("keys_to_act", [keys])[0]

    def _on_key_event(
            self, window, key: int, scancode: int, action: int, mode: int
        ) -> None:
            name = _key_to_name.get(key)
            if action == glfw.PRESS:
                self._keys_pressed.add(name)
                self._keys_clicked.add(name)
            elif action == glfw.RELEASE:
                if name in self._keys_pressed:
                    # hitting "fn" on a mac only seems to produce the RELEASE action
                    self._keys_pressed.remove(name)

    def iact(self):
        self.act(self.keys_to_act(self._keys_pressed))
