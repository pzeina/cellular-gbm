import numpy as np

# Define the size of the terrain embedding and spatial transcripto
terrain_embedding_size = (512, 512, 5)
spatial_transcripto_size = (512, 512, 7)

# Generate random terrain embedding and spatial transcripto
terrain_embedding = np.random.rand(*terrain_embedding_size)
np.save('tissue_env/envs/terrain_embedding.npy', terrain_embedding)

spatial_transcripto = np.random.rand(*spatial_transcripto_size)
np.save('tissue_env/envs/spatial_transcripto.npy', spatial_transcripto)