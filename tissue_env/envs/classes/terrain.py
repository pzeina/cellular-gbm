import argparse
import csv
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import torch


class TerrainEmbedding:
    """Embedding for terrain description."""

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        pass

    def encode(self) -> torch.Tensor:
        """Encode the terrain embedding into a tensor."""
        pass



class SpatialTranscripto:
    """Transcripto for spatial information."""

    def get_color(self) -> tuple[int, int, int]:
        """Return RGB color for rendering."""
        pass

    def encode(self) -> torch.Tensor:
        """Encode the spatial transcripto into a tensor."""
        pass



class TerrainTile:
    """Tile for terrain description."""

    terrain_embedding: TerrainEmbedding # HnE
    spatial_transcripto: SpatialTranscripto # ~ Agent

    def get_color_embedding(self) -> tuple[int, int, int]:
        """Return RGB color for rendering the terrain embedding."""
        return self.terrain_embedding.get_color()
    
    def get_color_transcripto(self) -> tuple[int, int, int]:
        """Return RGB color for rendering the spatial transcripto."""
        return self.spatial_transcripto.get_color()

    def get_embedding(self) -> TerrainEmbedding:
        """Get the embedding of the terrain."""
        return self.terrain_embedding

    def get_spatial_transcripto(self) -> SpatialTranscripto:
        """Get the spatial transcripto of the terrain."""
        return self.spatial_transcripto
    
    def encode(self) -> torch.Tensor:
        """Encode the terrain tile into a tensor."""
        return torch.cat([self.terrain_embedding.encode(), self.spatial_transcripto.encode()], dim=0)

class GameMap:
    """Represents the game's terrain map."""

    def __init__(self, terrain_embedding: TerrainEmbedding, spatial_transcripto: SpatialTranscripto, params: tuple[int, int, int]) -> None:
        """Initialize the game map with dimensions."""
        self.width: int = params[0]
        self.height: int = params[1]
        self.cell_size: int = params[2]
        self.terrain: np.ndarray = np.empty((self.height, self.width), dtype=TerrainTile)

        for y in range(self.height):
            for x in range(self.width):
                self.terrain[y, x] = TerrainTile(terrain_embedding[y,x], spatial_transcripto[y,x])

    def encode(self) -> torch.Tensor:
        """Generate a tensor with terrain properties for ML training."""
        zero_terrain_tile: TerrainTile = self.terrain[0, 0]
        zero_tile_tensor: torch.Tensor = zero_terrain_tile.encode()
        encoded_tile_length: int = zero_tile_tensor.size(0)

        encoded_map: torch.Tensor = torch.zeros((self.height, self.width, encoded_tile_length), dtype=torch.float32)
        for y in range(self.height):
            for x in range(self.width):
                terrain_tile: TerrainTile = self.terrain[y, x]
                if terrain_tile:
                    encoded_map[y, x] = terrain_tile.encode()
        return encoded_map

    def get_terrain_tile(self, x: int, y: int) -> TerrainTile:
        """Return the terrain tile at the specified coordinates."""
        return self.terrain[y, x]

    def render(self, screen: pygame.Surface, cell_size: int = 6) -> None:
        """Render the terrain map to a PyGame surface."""
        if not self.cell_size:
            self.cell_size = cell_size
        for y in range(self.height):
            for x in range(self.width):
                terrain_tile: TerrainTile = self.terrain[y, x]
                if terrain_tile:
                    pygame.draw.rect(
                        screen,
                        terrain_tile.get_color(),
                        (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size),
                    )


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
