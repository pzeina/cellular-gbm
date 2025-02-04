from gymnasium.envs.registration import register

register(
    id="tissue_env/TerrainWorld-v0",
    entry_point="tissue_env.envs:TerrainWorldEnv",
)
register(
    id="tissue_env/GridWorld-v0",
    entry_point="tissue_env.envs:GridWorldEnv",
)
