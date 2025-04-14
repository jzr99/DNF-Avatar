import torch

from . import _C
from .volumes import IrradianceVolumes

@torch.no_grad()
def recon_occlusion(
    H: int,
    W: int,
    bound: float,
    points: torch.Tensor,  # [HW, 3]
    normals: torch.Tensor,  # [HW, 3]
    occlusion_coefficients: torch.Tensor,
    occlusion_ids: torch.Tensor,
    aabb: torch.Tensor,
    sample_rays: int = 256,
    degree: int = 4,
    shift_ratio: float = 0.01,
) -> torch.Tensor:
    occlu_res = occlusion_ids.shape[0]
    half_grid = bound / float(occlu_res)

    # import pdb; pdb.set_trace()
    shift_points = points + normals * half_grid * shift_ratio
    # shift_points = points
    shift_points = shift_points.clamp(min=-bound, max=bound)
    (
        coefficients,  # [HW, d2, 1]
        coeff_ids,  # [HW, 8]
    ) = _C.sparse_interpolate_coefficients(
        occlusion_coefficients,
        occlusion_ids,
        aabb,
        shift_points,
        normals,
        degree,
    )
    coefficients = coefficients.permute(0, 2, 1)  # [HW, 1, d2]

    roughness = torch.ones([H * W, 1], dtype=torch.float32).cuda()
    occlusion = _C.SH_reconstruction(
        coefficients, normals, roughness, sample_rays, degree
    )  # [HW, 1]

    return occlusion

@torch.no_grad()
def recon_occlusion_1spp(
    H: int,
    W: int,
    bound: float,
    points: torch.Tensor,  # [HW, 3]
    normals: torch.Tensor,  # [HW, 3]
    occlusion_coefficients: torch.Tensor,
    occlusion_ids: torch.Tensor,
    aabb: torch.Tensor,
    sample_rays: int = 1,
    degree: int = 4,
    shift_ratio: float = 0.01,
) -> torch.Tensor:
    occlu_res = occlusion_ids.shape[0]
    half_grid = bound / float(occlu_res)

    # import pdb; pdb.set_trace()
    shift_points = points + normals * half_grid * shift_ratio
    # shift_points = points
    shift_points = shift_points.clamp(min=-bound, max=bound)
    (
        coefficients,  # [HW, d2, 1]
        coeff_ids,  # [HW, 8]
    ) = _C.sparse_interpolate_coefficients(
        occlusion_coefficients,
        occlusion_ids,
        aabb,
        shift_points,
        normals,
        degree,
    )
    coefficients = coefficients.permute(0, 2, 1)  # [HW, 1, d2]

    # coefficients = coefficients / 3.1415926

    roughness = torch.zeros([H * W, 1], dtype=torch.float32).cuda()
    occlusion = _C.SH_reconstruction(
        coefficients, normals, roughness, sample_rays, degree
    )  # [HW, 1]

    return occlusion


__all__ = ["_C", "recon_occlusion", "IrradianceVolumes", "recon_occlusion_1spp"]
