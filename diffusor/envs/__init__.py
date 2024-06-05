# Register environment classes here
# Register the environments
from gym.envs import register

from .base import EmptyEnv

from .metaworld import MetaWorldSawyerEnv, MetaWorldSawyerImageWrapper

try:
    from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

    for env_name in ALL_V2_ENVIRONMENTS.keys():
        ID = f"mw_{env_name}"
        register(id=ID, entry_point="diffusor.envs.metaworld:MetaWorldSawyerEnv", kwargs={"env_name": env_name})
        id_parts = ID.split("-")
        id_parts[-1] = "image-" + id_parts[-1]
        ID = "-".join(id_parts)
        register(id=ID, entry_point="diffusor.envs.metaworld:get_mw_image_env", kwargs={"env_name": env_name})
except ImportError:
    print("[research] Warning: Could not import MetaWorld Environments.")
