from gymnasium.envs.registration import register

# register(
#     id="tissue_env/TerrainWorld-v0",
#     entry_point="tissue_env.envs:TerrainWorldEnv",
# )
register(
    id="tissue_env/GridWorld-v0",
    entry_point="tissue_env.envs:GridWorldEnv",
)


import numpy as np
from gymnasium.envs.registration import register
from tissue_env.envs.terrain_world import TerrainWorldEnv
from tissue_env.envs.classes.terrain import TerrainEmbedding, SpatialTranscripto

def load_terrain_embedding():
    terrain_embedding = np.load('tissue_env/envs/data/embeddings_30a_v2.npy')
    return terrain_embedding

def load_spatial_transcripto():
    spatial_transcripto = np.load('tissue_env/envs/data/spatial_transcripto.npy')
    return spatial_transcripto

def register_terrain_world_env():
    terrain_embedding = load_terrain_embedding()
    size = terrain_embedding.shape
    print(terrain_embedding.shape)
    spatial_transcripto = np.random.rand(*size)
    map_params = (size[0], size[1], size[0])

    # spatial_transcripto = load_spatial_transcripto()

    register(
        id='tissue_env/TerrainWorld-v0',
        entry_point='tissue_env.envs:TerrainWorldEnv',
        kwargs={
            'terrain_embedding': terrain_embedding,
            'spatial_transcripto': spatial_transcripto,
            'map_params': map_params,
            'scores_file': 'tissue_env/envs/data/scores.npy',
            'render_mode': 'human'
        }
    )

register_terrain_world_env()