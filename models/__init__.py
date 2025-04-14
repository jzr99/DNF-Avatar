models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def make(name, config):
    model = models[name](config)
    return model


# IA model
from . import (
        # neus_ir_avatar,
    neus_vol_ir_avatar,
)

# Radiance field modules
from .rf import geometry
from .rf import radiance
from .rf import density

# Pose modules
from .pose import pose_encoder
from .pose import pose_correction

# PBR modules
from .pbr import material

# Articulation modules
from .deformers import deformer
from .deformers import non_rigid_deformer

# Import external PBR classes
import lib.pbr

EnvironmentLightTensor = register("envlight-tensor")(lib.pbr.EnvironmentLightTensor)
EnvironmentLightSG = register("envlight-SG")(lib.pbr.EnvironmentLightSG)
EnvironmentLightMLP = register("envlight-mlp")(lib.pbr.EnvironmentLightMLP)
EnvironmentLightNGP = register("envlight-ngp")(lib.pbr.EnvironmentLightNGP)
Mirror = register("brdf-mirror")(lib.pbr.Mirror)
Lambertian = register("brdf-lambertian")(lib.pbr.Lambertian)
GGX = register("brdf-ggx")(lib.pbr.GGX)
DiffuseSGGX = register("phase-diffuse-sggx")(lib.pbr.DiffuseSGGX)
SpecularSGGX = register("phase-specular-sggx")(lib.pbr.SpecularSGGX)
MultiLobe = register("brdf-multi-lobe")(lib.pbr.MultiLobe)
MultiLobeSGGX = register("phase-multi-lobe")(lib.pbr.MultiLobeSGGX)
