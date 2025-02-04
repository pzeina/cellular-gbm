from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import tissue_env.envs.classes.terrain as terrain
from pathlib import Path
import torch.nn as nn


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class TerrainWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, array_embedding: np.ndarray, array_transcripto: np.ndarray, map_params: tuple[int, int, int], render_mode=None, max_steps=100):
        """Initialize the environment."""

        self.size = map_params[2]  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps  # Maximum number of steps in the simulation

        # Initialize the game map
        self.game_map = terrain.GameMap(array_embedding, array_transcripto, map_params)

        # Observations are dictionaries with the agent's and the path's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.size - 1, shape=(2,), dtype=int),
                "path": spaces.Sequence(spaces.Box(0, self.size - 1, shape=(2,), dtype=int))
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.path = []
        self.steps = 0

    def _get_obs(self):
        return {"agent": self._agent_location, "path": np.array(self.path)}

    def _get_info(self):
        return {
            "similarity": nn.functional.mse_loss(
                self.game_map.terrain[self._agent_location[1], self._agent_location[0]].encode(),
                self.game_map.terrain[self.path[-2][1], self.path[-2][0]].encode()
            ) if len(self.path) > 1 else 0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.path = [self._agent_location.copy()]
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        self.path.append(self._agent_location.copy())
        self.steps += 1

        reward = -nn.functional.mse_loss(
            self.game_map.terrain[self._agent_location[1], self._agent_location[0]].encode(),
            self.game_map.terrain[self.path[-2][1], self.path[-2][0]].encode()
        ) if len(self.path) > 1 else 0

        terminated = self.steps >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        pix_square_size = self.window_size / self.size

        self.game_map.render(canvas, cell_size=pix_square_size)

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 2,
        )

        for pos in self.path:
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (pos + 0.5) * pix_square_size,
                pix_square_size / 4,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
