from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import tissue_env.envs.classes.terrain as terrain
from pathlib import Path



class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class TerrainWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 64, target_zone_size: int = 8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.target_zone_size = target_zone_size  # Size of the target zone

        # Load the terrain map
        self.game_map  = terrain.GameMap.load_from_csv(Path(__file__).parent / "data" / "terrain.csv")

        print(self.game_map.terrain[0,0])

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target_zone": spaces.Box(0, size - 1, shape=(2, 2), dtype=int),
            }
        )


        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target_zone": self._target_zone}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_zone_center, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Define the target zone
        self._target_zone_center = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_zone = np.array([
            self._target_zone_center - self.target_zone_size // 2,
            self._target_zone_center + self.target_zone_size // 2
        ])
        self._target_zone = np.clip(self._target_zone, 0, self.size - 1)

        # Choose the agent's location uniformly at random and ensure it is not inside the target zone
        while True:
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not (self._target_zone[0] <= self._agent_location).all() or not (self._agent_location <= self._target_zone[1]).all():
                break

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.all(self._target_zone[0] <= self._agent_location) and np.all(self._agent_location <= self._target_zone[1])
        reward = 1 if terminated else 0  # Binary sparse rewards
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
        # canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Render the game map
        self.game_map.render(canvas, cell_size=pix_square_size)

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 2,
        )

        # Draw the target zone
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self._target_zone[0] * pix_square_size,
                (self.target_zone_size * pix_square_size, self.target_zone_size * pix_square_size),
            ),
            width=int(pix_square_size)
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
