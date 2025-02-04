from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import tissue_env.envs.classes.terrain as terrain
from pathlib import Path
import torch.nn as nn
from tissue_env.envs.classes.terrain import TerrainTile


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class TerrainWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, terrain_embedding: np.ndarray, spatial_transcripto: np.ndarray, map_params: tuple[int, int, int], scores_file: str, render_mode=None, max_steps=1000):
        """Initialize the environment."""

        self.size = map_params[2]  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.max_steps = max_steps  # Maximum number of steps in the simulation

        # Load the scores from the given file
        color_array: np.ndarray = np.load(scores_file)

        # Initialize the game map
        self.game_map = terrain.GameMap(terrain_embedding, spatial_transcripto, map_params, color_array)

        # Observations are dictionaries with the agent's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, self.size - 1, shape=(2,), dtype=int)

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

        # Initialize the agent's identity from a random cell in the terrain embedding
        random_y = np.random.randint(0, terrain_embedding.shape[0])
        random_x = np.random.randint(0, terrain_embedding.shape[1])
        self.agent_identity = terrain.TerrainEmbedding(terrain_embedding[random_y, random_x])

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        current_position: TerrainTile = self.game_map.get_terrain_tile(*self._agent_location)
        return {
            "similarity": nn.functional.mse_loss(
                current_position.encode_embedding(),
                self.agent_identity.encode()
            ).item()
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
        new_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if the new location is valid
        new_tile: TerrainTile = self.game_map.terrain[new_location[1], new_location[0]]
        if not self.is_valid_tile(new_tile):
            reward = -1  # Penalize for attempting to move to an invalid tile
            terminated = False
            observation = self._get_obs()
            info = self._get_info()
            return observation, reward, terminated, False, info

        self._agent_location = new_location
        self.steps += 1

        current_tile: TerrainTile = self.game_map.terrain[self._agent_location[1], self._agent_location[0]]
        similarity = nn.functional.mse_loss(current_tile.encode_embedding(), self.agent_identity.encode()).item()

        # Convert path to a set of tuples for efficient membership checking
        path_set = {tuple(loc) for loc in self.path}

        # Calculate the reward
        if tuple(self._agent_location) in path_set:
            reward = -1  # Negative reward for revisiting tiles
        else:
            reward = -similarity  # Positive reward for exploring different tiles

        self.path.append(self._agent_location.copy())

        terminated = self.steps >= self.max_steps
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def is_valid_tile(self, tile: TerrainTile) -> bool:
        """Check if the tile is valid."""
        # Implement your logic to check if the tile is valid
        # For example, you can check if the tile's embedding or transcripto values are within a valid range
        return True  # Replace with actual validation logic

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
