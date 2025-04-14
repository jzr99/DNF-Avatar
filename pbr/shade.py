import os
from typing import Dict, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from .light import CubemapLight


# Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II"
# https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
        torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)]
        + r[..., (1,)]
    )
    AB = (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)
    return AB


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


# Tone Mapping
def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)


def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


# def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
#     return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)
#
# def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
#     assert f.shape[-1] == 3 or f.shape[-1] == 4
#     out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
#     for i in range(len(out.shape)):
#         assert out.shape[i] == f.shape[i]
#     return out

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _rgb_to_srgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = (
        torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1)
        if f.shape[-1] == 4
        else _srgb_to_rgb(f)
    )
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def get_brdf_lut() -> torch.Tensor:
    brdf_lut_path = os.path.join(os.path.dirname(__file__), "brdf_256_256.bin")
    brdf_lut = torch.from_numpy(
        np.fromfile(brdf_lut_path, dtype=np.float32).reshape(1, 256, 256, 2)
    )
    return brdf_lut

def line_to_point(intersection_point, ref_dir, eps):
    N = intersection_point.shape[0]
    diff_B_C = ref_dir # [N, 3]
    diff_A_C = intersection_point[:, None] - intersection_point[None, :] # [N, N, 3]
    # calculate dot product
    dot_result = torch.sum(diff_B_C[None, :] * diff_A_C, dim=-1) # [N, N]
    # calculate mask
    mask = dot_result > eps

    norm_lines = torch.norm(diff_B_C, dim=-1)
    cross_result = torch.cross(diff_B_C[None, :].expand(N, -1, -1), diff_A_C, dim=-1)
    norm_cross = torch.norm(cross_result, dim=-1)
    D = norm_cross / norm_lines
    D[~mask] = 999999999

    return D

def flip_normals(pc, camera_center):
    unsigned_normal = pc.get_normal.clone()
    # camera_center = c2w[:3, 3]
    deformed_position = pc.get_xyz
    norm_view_dir = (camera_center - deformed_position) / (
                torch.norm(camera_center - deformed_position, dim=-1, keepdim=True) + 1e-8)
    unsigned_normal[(norm_view_dir * unsigned_normal).sum(dim=-1) < -0.3] = -unsigned_normal[
        (norm_view_dir * unsigned_normal).sum(dim=-1) < -0.3]

    return unsigned_normal

def pbr_shading_gs(deformed_gaussian,
                   camera_center,
                  light,
                  occlusion,  # [pc, 1]
                  irradiance,  # [pc, 1]
                  brdf_lut,
                   three_channel_ratio=None):
    pc = deformed_gaussian
    xyz = pc.get_xyz
    camera_center = camera_center.reshape(1, 3)
    view_dirs = (camera_center - xyz) / torch.norm(camera_center - xyz, dim=-1, keepdim=True)
    normal = pc.get_normal
    normal = flip_normals(pc, camera_center)
    albedo = pc.get_albedo
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    H = normal.shape[0]
    W = 1
    # prepare
    normals = normal.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)
    metallic = metallic.reshape(1, H, W, 1)
    occlusion = occlusion.reshape(H, W, 1)
    irradiance = irradiance.reshape(H, W, 1)

    if three_channel_ratio is not None:
        albedo = albedo * three_channel_ratio.reshape(1, 1, 1, 3)

    results = {}
    # prepare
    ref_dirs = (
            2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    # Diffuse lookup
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    diffuse_rgb = diffuse_light * albedo  # [1, H, W, 3]

    diffuse_rgb = diffuse_rgb * (1 - metallic)

    # specular
    NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    # IA
    # F0 = (0.04 * (1.0 - metallic) + kd * metallic)

    # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1, H, W, 3]
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    if occlusion is not None:
        specular_rgb = specular_rgb * occlusion[None]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

    render_rgb = render_rgb.reshape(H, 3)  # [H, 3]
    # TODO change render_rgb after rasterization (done)
    # render_rgb = linear_to_srgb(render_rgb)
    return render_rgb


def pbr_shading_gs_linepoint(deformed_gaussian,
                   camera_center,
                  light,
                  occlusion,  # [pc, 1]
                  irradiance,  # [pc, 1]
                  brdf_lut,):
    pc = deformed_gaussian
    xyz = pc.get_xyz
    camera_center = camera_center.reshape(1, 3)
    view_dirs = (camera_center - xyz) / torch.norm(camera_center - xyz, dim=-1, keepdim=True)
    normal = pc.get_normal
    albedo = pc.get_albedo
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    H = normal.shape[0]
    W = 1
    # prepare
    normals = normal.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)
    metallic = metallic.reshape(1, H, W, 1)
    occlusion = occlusion.reshape(H, W, 1)
    irradiance = irradiance.reshape(H, W, 1)

    results = {}
    # prepare
    ref_dirs = (
            2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    import pdb;pdb.set_trace()
    D = line_to_point(xyz, ref_dirs.reshape(-1, 3), 1e-2)

    # Diffuse lookup
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    diffuse_rgb = diffuse_light * albedo  # [1, H, W, 3]

    diffuse_rgb = diffuse_rgb * (1 - metallic)

    # specular
    NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    # IA
    # F0 = (0.04 * (1.0 - metallic) + kd * metallic)

    # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1, H, W, 3]
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    if occlusion is not None:
        specular_rgb = specular_rgb * occlusion[None]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

    render_rgb = render_rgb.reshape(H, 3)  # [H, 3]
    # TODO change render_rgb after rasterization (done)
    # render_rgb = linear_to_srgb(render_rgb)
    return render_rgb



def pbr_shading(
    light: CubemapLight,
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    tone: bool = False,
    gamma: bool = False,
    occlusion: Optional[torch.Tensor] = None,  # [H, W, 1]
    irradiance: Optional[torch.Tensor] = None,  # [H, W, 1]
    metallic: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    return_background: bool = True,
) -> Dict:
    H, W, _ = normals.shape
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # prepare
    normals = normals.reshape(1, H, W, 3)
    view_dirs = view_dirs.reshape(1, H, W, 3)
    albedo = albedo.reshape(1, H, W, 3)
    roughness = roughness.reshape(1, H, W, 1)

    # formulate roughness
    # rmax, rmin = 1.0, 0.8
    # roughness = roughness * (rmax - rmin) + rmin

    results = {}
    # prepare
    ref_dirs = (
        2.0 * (normals * view_dirs).sum(-1, keepdims=True).clamp(min=0.0) * normals - view_dirs
    )  # [1, H, W, 3]

    # Diffuse lookup
    diffuse_light = dr.texture(
        light.diffuse[None, ...],  # [1, 6, 16, 16, 3]
        normals.contiguous(),  # [1, H, W, 3]
        filter_mode="linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion[None] + (1 - occlusion[None]) * irradiance[None]

    results["diffuse_light"] = diffuse_light[0]
    diffuse_rgb = diffuse_light * albedo  # [1, H, W, 3]

    diffuse_rgb = diffuse_rgb * (1-metallic)

    # specular
    NoV = saturate_dot(normals, view_dirs)  # [1, H, W, 1]
    fg_uv = torch.cat((NoV, roughness), dim=-1)  # [1, H, W, 2]
    fg_lookup = dr.texture(
        brdf_lut,  # [1, 256, 256, 2]
        fg_uv.contiguous(),  # [1, H, W, 2]
        filter_mode="linear",
        boundary_mode="clamp",
    )  # [1, H, W, 2]

    # Roughness adjusted specular env lookup
    miplevel = light.get_mip(roughness)  # [1, H, W, 1]
    spec = dr.texture(
        light.specular[0][None, ...],  # [1, 6, env_res, env_res, 3]
        ref_dirs.contiguous(),  # [1, H, W, 3]
        mip=list(m[None, ...] for m in light.specular[1:]),
        mip_level_bias=miplevel[..., 0],  # [1, H, W]
        filter_mode="linear-mipmap-linear",
        boundary_mode="cube",
    )  # [1, H, W, 3]

    # IA
    # F0 = (0.04 * (1.0 - metallic) + kd * metallic)

    # Compute aggregate lighting
    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [1, H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic
    reflectance = F0 * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]  # [1, H, W, 3]
    specular_rgb = spec * reflectance  # [1, H, W, 3]

    if occlusion is not None:
        specular_rgb = specular_rgb * occlusion[None]

    render_rgb = diffuse_rgb + specular_rgb  # [1, H, W, 3]

    render_rgb = render_rgb.squeeze()  # [H, W, 3]

    if tone:
        # Tone Mapping
        render_rgb = aces_film(render_rgb)
    else:
        render_rgb = render_rgb.clamp(min=0.0, max=1.0)

    # render_rgb = torch.where(mask>0.93, render_rgb, background)
    # render_rgb = render_rgb * mask + background * (1 - mask)

    # background lookup
    if return_background:
        background_light = dr.texture(
            light.base[None, ...],  # [1, 6, 256, 256, 3]
            -view_dirs.contiguous(),  # [1, H, W, 3]
            filter_mode="linear",
            boundary_mode="cube",
        )  # [1, H, W, 3]
        # import pdb;pdb.set_trace()

        render_rgb_background = render_rgb * mask + background_light * (1 - mask)
        diffuse_rgb_background = diffuse_rgb * mask + background_light * (1 - mask)
        # render_rgb_background = torch.where(mask, render_rgb, background_light.squeeze())
        # diffuse_rgb_background = torch.where(mask, diffuse_rgb, background_light.squeeze())

        ### NOTE: close `gamma` will cause better resuls in novel view synthesis but wrose relighting results.
        ### NOTE: it is worth to figure out a better way to handle both novel view synthesis and relighting
        if gamma:
            render_rgb = linear_to_srgb(render_rgb.squeeze())
            render_rgb_background = linear_to_srgb(render_rgb_background.squeeze())
            diffuse_rgb_background = linear_to_srgb(diffuse_rgb_background.squeeze())
            diffuse_rgb = linear_to_srgb(diffuse_rgb.squeeze())
            specular_rgb = linear_to_srgb(specular_rgb.squeeze())
            background_light = linear_to_srgb(background_light.squeeze())

        render_rgb = render_rgb * mask + background * (1 - mask)

        results.update(
            {
                "render_rgb": render_rgb.squeeze(),
                "diffuse_rgb": diffuse_rgb.squeeze(),
                "specular_rgb": specular_rgb.squeeze(),
                "background_light": background_light.squeeze(),
                "render_rgb_background": render_rgb_background.squeeze(),
                "diffuse_rgb_background": diffuse_rgb_background.squeeze(),
            }
        )
    else:
        if gamma:
            render_rgb = linear_to_srgb(render_rgb.squeeze())
        # import pdb;pdb.set_trace()
        render_rgb = render_rgb.reshape(H, W, 3) * mask.reshape(H, W, 1) + background * (1 - mask.reshape(H, W, 1))
        results.update(
            {
                "render_rgb": render_rgb.squeeze(),
            }
        )


    return results



def pbr_shading_glossy(
    emitter,
    normals: torch.Tensor,  # [H, W, 3]
    dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    tone: bool = False,
    gamma: bool = False,
    occlusion: Optional[torch.Tensor] = None,  # [H, W, 1]
    irradiance: Optional[torch.Tensor] = None,  # [H, W, 1]
    metallic: Optional[torch.Tensor] = None,
    brdf_lut: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict:
    # TODO check dir & normal is normalized
    # brdf_lut should be read from new file
    H, W = normals.shape[0], normals.shape[1]
    normals = normals.reshape(H*W, 3)
    dirs = dirs.reshape(H*W, 3)
    albedo = albedo.reshape(H*W, 3)
    roughness = roughness.reshape(H*W, 1)
    mask = mask.reshape(H*W, 1)
    metallic = metallic.reshape(H*W, 1)

    wi = dirs
    wo = torch.sum(wi * normals, -1, keepdim=True) * normals * 2 - wi
    NoV = torch.sum(normals * wi, -1, keepdim=True)

    diffuse_albedo = (1 - metallic) * albedo
    diffuse_light = emitter.eval_mip(normals)

    diff_rgb_pbr = diffuse_albedo * diffuse_light

    specular_albedo = 0.04 * (1 - metallic) + metallic * albedo
    specular_light = emitter.eval_mip(wo, specular=True, roughness=roughness)

    fg_uv = torch.cat([torch.clamp(NoV, min=0.0, max=1.0), torch.clamp(roughness, min=0.0, max=1.0)], -1)
    pn, bn = dirs.shape[0], 1
    fg_lookup = dr.texture(brdf_lut, fg_uv.reshape(1, pn // bn, bn, fg_uv.shape[-1]).contiguous(),
                           filter_mode='linear',
                           boundary_mode='clamp').reshape(pn, 2)
    specular_ref = (specular_albedo * fg_lookup[:, 0:1] + fg_lookup[:, 1:2])
    spec_rgb_pbr = specular_ref * specular_light

    render_rgb = diff_rgb_pbr + spec_rgb_pbr
    if gamma:
        render_rgb = linear_to_srgb(render_rgb.squeeze())
        diff_rgb_pbr = linear_to_srgb(diff_rgb_pbr.squeeze())
        spec_rgb_pbr = linear_to_srgb(spec_rgb_pbr.squeeze())
    # import pdb;pdb.set_trace()
    render_rgb = render_rgb * mask + background * (1 - mask)
    diff_rgb_pbr = diff_rgb_pbr * mask + background * (1 - mask)
    spec_rgb_pbr = spec_rgb_pbr * mask + background * (1 - mask)

    results = {}
    results.update(
        {
            "render_rgb": render_rgb.reshape(H, W, 3),
            "diffuse_rgb": diff_rgb_pbr.reshape(H, W, 3),
            "specular_rgb": spec_rgb_pbr.reshape(H, W, 3),
        }
    )
    return results



