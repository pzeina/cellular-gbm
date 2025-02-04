import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class TerrainEmbedding(np.ndarray):
    """Embedding for terrain description."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        # Define a transformation matrix to map the vector to RGB space
        transformation_matrix = np.random.rand(3, self.shape[-1])  # Random 3xN matrix

        # Apply the transformation
        rgb_vector = np.dot(transformation_matrix, self)

        # Clip the values to the range [0, 1] and scale to [0, 255]
        rgb_vector = np.clip(rgb_vector, 0, 1) * 255

        # Convert to integer and return as a tuple
        return tuple(rgb_vector.astype(int))

    def encode(self) -> torch.Tensor:
        """Encode the terrain embedding into a tensor."""
        return torch.tensor(self, dtype=torch.float32)


class SpatialTranscripto(np.ndarray):
    """Transcripto for spatial information."""

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        # Define a transformation matrix to map the vector to RGB space
        transformation_matrix = np.random.rand(3, self.shape[-1])  # Random 3xN matrix

        # Apply the transformation
        rgb_vector = np.dot(transformation_matrix, self)

        # Clip the values to the range [0, 1] and scale to [0, 255]
        rgb_vector = np.clip(rgb_vector, 0, 1) * 255

        # Convert to integer and return as a tuple
        return tuple(rgb_vector.astype(int))

    def encode(self) -> torch.Tensor:
        """Encode the spatial transcripto into a tensor."""
        return torch.tensor(self, dtype=torch.float32)


class TerrainTile:
    """Tile for terrain description."""

    def __init__(self, terrain_embedding: TerrainEmbedding, spatial_transcripto: SpatialTranscripto, color: tuple[int, int, int]) -> None:
        self.terrain_embedding = terrain_embedding
        self.spatial_transcripto = spatial_transcripto
        self.color = color


    def get_color_embedding(self) -> tuple[int, int, int]:
        return self.terrain_embedding.get_color()

    def get_color_transcripto(self) -> tuple[int, int, int]:
        return self.spatial_transcripto.get_color()

    def get_embedding(self) -> TerrainEmbedding:
        return self.terrain_embedding

    def get_spatial_transcripto(self) -> SpatialTranscripto:
        return self.spatial_transcripto

    def get_color(self) -> tuple[int, int, int]:
        return self.color

    def encode(self) -> torch.Tensor:
        return torch.cat((self.terrain_embedding.encode(), self.spatial_transcripto.encode()), dim=0)
    
    def encode_embedding(self) -> torch.Tensor:
        return self.terrain_embedding.encode()


class GameMap:
    """Represents the game's terrain map."""

    def __init__(self, array_embedding: np.ndarray, array_transcripto: np.ndarray, params: tuple[int, int, int], color_array: np.ndarray) -> None:
        self.width, self.height, _ = params
        self.terrain = np.empty((self.height, self.width), dtype=object)

        # Use a custom colormap for heatmap coloring
        colormap = cm.get_cmap('coolwarm')

        for y in range(self.height):
            for x in range(self.width):
                # Assuming color_array contains float values in the range [0, 1]
                color = colormap(color_array[y, x])[:3]  # Get RGB values from colormap
                color = tuple((np.array(color) * 255).astype(int))  # Scale to [0, 255] and convert to int
                self.terrain[y, x] = TerrainTile(TerrainEmbedding(array_embedding[y, x]), SpatialTranscripto(array_transcripto[y, x]), color)

    def encode(self) -> torch.Tensor:
        encoded_tiles = [self.terrain[y, x].encode() for y in range(self.height) for x in range(self.width)]
        return torch.stack(encoded_tiles).view(self.height, self.width, -1)

    def get_terrain_tile(self, x: int, y: int) -> TerrainTile:
        return self.terrain[y, x]

    def render(self, screen: pygame.Surface, cell_size: int = 6) -> None:
        for y in range(self.height):
            for x in range(self.width):
                tile: TerrainTile = self.terrain[y, x]
                color = tile.get_color()
                pygame.draw.rect(screen, color, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))


class GameRenderer:
    """Handles the game rendering and PyGame setup."""

    def __init__(self, game_map: GameMap) -> None:
        """Initialize the renderer with the game map."""
        pygame.init()
        self.game_map = game_map
        self.screen_width = game_map.width * game_map.cell_size
        self.screen_height = game_map.height * game_map.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Terrain Game")
        self.clock = pygame.time.Clock()

    def run(self) -> None:
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save map when 'S' is pressed
                        self.game_map.save_to_csv(Path("map.csv"))
                    else:
                        pass

            self.screen.fill((0, 0, 0))
            self.game_map.render(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main() -> None:
    """Example usage of the terrain system."""
    parser = argparse.ArgumentParser(description="Terrain system")
    parser.add_argument("--load", type=str, help="Load the map from the specified file")
    args = parser.parse_args()

    if args.load:
        # Load the map from the specified file
        loaded_map = GameMap.load_from_csv(args.load)
    else:
        # Create new random map
        game_map = GameMap(100, 100)
        game_map.generate_random_map()

        # Save the map
        maps_dir = Path(__file__).resolve().parent / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)

        map_path = maps_dir / "example_map.csv"
        game_map.save_to_csv(map_path)

        # Load the map
        loaded_map = GameMap.load_from_csv(map_path)

    # Visualize the loaded map
    renderer = GameRenderer(loaded_map)
    renderer.run()


if __name__ == "__main__":
    main()
