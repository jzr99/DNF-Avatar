#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
# from gaussian_renderer_pbr import render, render_fast
from gaussian_renderer_2dgs import render as render_2dgs
from gaussian_renderer_2dgs import render_fast as render_fast_2dgs
import torchvision
from utils.general_utils import fix_random
# from scene import GaussianModel
from scene.gaussian_model_2dgs import GaussianModel
import torch.nn.functional as F

from utils.general_utils import Evaluator, PSEvaluator, RANAEvaluator

import hydra
from omegaconf import OmegaConf
import wandb

from typing import Dict, List, Optional, Tuple, Union
import nvdiffrast.torch as dr
import kornia
from pbr import get_brdf_lut, pbr_shading, pbr_shading_gs, linear_to_srgb
from utils.camera_utils import get_world_camera_rays_from_intrinsic
from utils.image_utils import turbo_cmap
from gs_ir import recon_occlusion_1spp, IrradianceVolumes
import cv2
from utils.general_utils import build_rotation
from models.utils import compute_albedo_rescale_factor


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb
def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

# def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
#     cubemap = torch.zeros(
#         6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
#     )
#     for s in range(6):
#         gy, gx = torch.meshgrid(
#             torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
#             torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
#             indexing="ij",
#         )
#         v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
#
#         tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
#         tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
#         texcoord = torch.cat((tu, tv), dim=-1)
#
#         cubemap[s, ...] = dr.texture(
#             latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
#         )[0]
#     return cubemap

def flip_normals(pc, c2w):
    unsigned_normal = pc.get_normal.clone()
    camera_center = c2w[:3, 3]
    deformed_position = pc.get_xyz
    norm_view_dir = (camera_center - deformed_position) / (
                torch.norm(camera_center - deformed_position, dim=-1, keepdim=True) + 1e-8)
    unsigned_normal[(norm_view_dir * unsigned_normal).sum(dim=-1) < -0.3] = -unsigned_normal[
        (norm_view_dir * unsigned_normal).sum(dim=-1) < -0.3]

    return unsigned_normal
def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
        v[..., 2] = -v[..., 2]

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        # tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        tv = torch.asin(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi + 0.5
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap

def transform_normals(w2c, normals):
    """ Convert world-space normal map into OpenGL camera space
    """
    H, W = normals.shape[1], normals.shape[2]
    normals = normals.reshape(3, H*W).transpose(1, 0)
    # Convert to camera space, if necessary
    normals = torch.matmul(normals, w2c[:3, :3].transpose(0, 1))

    # Convert OpenCV to OpenGL convention
    normals = normals * torch.tensor([1.0, -1.0, -1.0], device=normals.device)
    normals = normals.transpose(1, 0).reshape(3, H, W)

    return normals

def test(config):
    with torch.no_grad():
        rendering_type = config.rendering_type  # 'forward_pbr' or 'diff_pbr'
        brdf_lut = get_brdf_lut().cuda()
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)
        scene.cubemap.export_envmap(os.path.join(config.exp_dir, "cubemap_256_origin.png"), [256,256])
        if config.get('hdri', None):
            print(f"read hdri from {config.hdri}")
            hdri = read_hdr(config.hdri)
            hdri = torch.from_numpy(hdri).cuda()
            res = scene.cubemap.base.data.shape[1]
            scene.cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
            scene.cubemap.export_envmap(os.path.join(config.exp_dir, "cubemap_256_relight.png"), [256, 256])
            scene.cubemap.export_envmap_IA(os.path.join(config.exp_dir, "cubemap_256_relight_IA.png"), [1024, 2048], hdri)

        scene.cubemap.build_mips()
        # import pdb;pdb.set_trace()
        # scene.gaussians.prune_small_opacity()
        # scene.gaussians.prune_large_condition_number()
        print('shape after filter!!', scene.gaussians.get_xyz.shape)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        depth_path = os.path.join(config.exp_dir, config.suffix, 'depth')
        makedirs(depth_path, exist_ok=True)

        normal_path = os.path.join(config.exp_dir, config.suffix, 'normal')
        makedirs(normal_path, exist_ok=True)

        normal_depth_path = os.path.join(config.exp_dir, config.suffix, 'normal_depth')
        makedirs(normal_depth_path, exist_ok=True)

        opacity_path = os.path.join(config.exp_dir, config.suffix, 'opacity')
        makedirs(opacity_path, exist_ok=True)

        occlusion_path = os.path.join(config.exp_dir, config.suffix, 'occlusion')
        makedirs(occlusion_path, exist_ok=True)

        render_pbr_path = os.path.join(config.exp_dir, config.suffix, 'renders_pbr')
        makedirs(render_pbr_path, exist_ok=True)

        albedo_path = os.path.join(config.exp_dir, config.suffix, 'albedo')
        makedirs(albedo_path, exist_ok=True)

        metallic_path = os.path.join(config.exp_dir, config.suffix, 'metallic')
        makedirs(metallic_path, exist_ok=True)

        roughness_path = os.path.join(config.exp_dir, config.suffix, 'roughness')
        makedirs(roughness_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        iter_start_lbs = torch.cuda.Event(enable_timing=True)
        iter_end_lbs = torch.cuda.Event(enable_timing=True)

        iter_start_occ = torch.cuda.Event(enable_timing=True)
        iter_end_occ = torch.cuda.Event(enable_timing=True)

        iter_start_shading = torch.cuda.Event(enable_timing=True)
        iter_end_shading = torch.cuda.Event(enable_timing=True)

        iter_start_rasterization = torch.cuda.Event(enable_timing=True)
        iter_end_rasterization = torch.cuda.Event(enable_timing=True)

        iter_start_diffuse = torch.cuda.Event(enable_timing=True)
        iter_end_diffuse = torch.cuda.Event(enable_timing=True)

        # evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()
        if config.dataset.name == 'people_snapshot':
            evaluator = PSEvaluator()
        elif config.dataset.name == 'rana' or config.dataset.name == 'synthetichuman':
            evaluator = RANAEvaluator()
        else:
            evaluator = Evaluator()
            raise NotImplementedError(f"Unknown dataset {config.dataset.name}")

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        times_diffuse = []
        times_lbs = []
        times_occ = []
        times_shading = []
        times_rasterization = []

        psnrs_pbr = []
        ssims_pbr = []
        lpipss_pbr = []

        normal_error_list = []
        psnrs_pbr_albedo = []
        ssims_pbr_albedo = []
        lpipss_pbr_albedo = []

        occlusion_flag = True
        occlusion_ids_list =[]
        occlusion_coefficients_list =[]
        points = scene.gaussians.get_xyz
        # select part group
        part_group = [
            [16],
            [18, 20, 22],
            [17],
            [19, 21, 23],
            [2, ],
            [5, 8, 11],
            [1, ],
            [4, 7, 10],
            # [2, 5, 8, 11],
            # [1, 4, 7, 10],
            [0, 3, 6, 9, 13, 14, 12, 15],
        ]
        points_group_one_hot = torch.zeros(points.shape[0], len(part_group), device=points.device)
        self_occlusion_list = []
        if occlusion_flag and (config.enable_occ_type == 'pixel' or config.enable_occ_type == 'gaussian'):
            for i in range(9):
                filepath = os.path.join(config.occ_dir, f"occlusion_volumes_{i}.pth")
                print(f"begin to load occlusion volumes from {filepath}")
                occlusion_volumes = torch.load(filepath)
                occlusion_ids_list.append(occlusion_volumes["occlusion_ids"])
                occlusion_coefficients_list.append(occlusion_volumes["occlusion_coefficients"])
                occlusion_degree = occlusion_volumes["degree"]
                bound = occlusion_volumes["bound"]
                aabb = torch.tensor(
                    [-bound, -bound, -bound, bound, bound, bound]).cuda()
            occlusion_flag = False
            # calculate self occ
            # TODO we can calculate it outside the loop
            points = scene.gaussians.get_xyz
            normal_points = scene.gaussians.get_canonical_normal
            # real_rotation = build_rotation(gaussians._rotation)
            # real_scales = scene.gaussians.get_scaling
            # dim1_index = torch.arange(0, real_scales.shape[0], device=real_scales.device)
            # small_index = torch.argmin(real_scales, dim=-1)
            # rotation_vec = real_rotation[dim1_index, :, small_index]
            # normal_points = rotation_vec / torch.norm(rotation_vec, dim=-1, keepdim=True)
            for i in range(9):
                occlusion = recon_occlusion_1spp(
                    H=points.shape[0],
                    W=1,
                    bound=bound,
                    points=points.clamp(min=-bound, max=bound).contiguous(),
                    normals=normal_points.contiguous(),
                    occlusion_coefficients=occlusion_coefficients_list[i],
                    occlusion_ids=occlusion_ids_list[i],
                    aabb=aabb,
                    degree=occlusion_degree,
                )
                # import pdb;pdb.set_trace()
                occlusion = occlusion.reshape(points.shape[0])
                self_occlusion_list.append(occlusion)

            for idx, part in enumerate(part_group):
                # mask different body part
                _, points_weights_smpl = scene.converter.deformer.rigid.query_weights(points)
                points_weights_part_idx = points_weights_smpl.argmax(dim=1)

                # select point based on the part group
                part = torch.tensor(part).cuda()
                point_mask = points_weights_part_idx.reshape(points_weights_part_idx.shape[0], 1) == part.reshape(1, -1)
                point_mask = point_mask.any(dim=1)
                points_group_one_hot[point_mask, idx] = 1
            self_occlusion_list = torch.stack(self_occlusion_list, dim=-1)  # [N, 9]
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            view = scene.test_dataset[idx]
            data = view
            c2w = torch.inverse(data.w2c_opencv)
            if data.data.get('hdri', None) is not None:
                # hdri = torch.from_numpy(hdri).cuda()
                res = scene.cubemap.base.data.shape[1]
                scene.cubemap.base.data = latlong_to_cubemap(data.hdri, [res, res])
            iter_start_diffuse.record()
            scene.cubemap.build_mips()  # build mip for environment light
            iter_end_diffuse.record()
            torch.cuda.synchronize()
            elapsed_diffuse = iter_start_diffuse.elapsed_time(iter_end_diffuse)
            times_diffuse.append(elapsed_diffuse)

            iter_start.record()

            iter_start_lbs.record()

            pc, loss_reg, colors_precomp = scene.convert_gaussians(data, config.opt.iterations, compute_loss=False, compute_color=False)

            iter_end_lbs.record()
            torch.cuda.synchronize()
            elapsed_lbs = iter_start_lbs.elapsed_time(iter_end_lbs)
            times_lbs.append(elapsed_lbs)

            iter_start_occ.record()
            if config.enable_occ_type == 'gaussian':
                # print("begin to use gaussian splatting to render occlusion")
                # deformed normal
                normal_points = pc.get_normal
                points = pc.get_xyz

                part_joint = [16, 18, 17, 19, 2, 5, 1, 4, 0]
                part_occlusion_list = []
                for i in range(9):
                    tfs = data.bone_transforms  # [B, 4, 4]
                    tfs_part = tfs[part_joint[i]]
                    n_pts = points.shape[0]
                    homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=points.device)
                    x_hat_homo = torch.cat([points, homo_coord], dim=-1).view(n_pts, 4, 1)
                    inv_tfs = torch.linalg.inv(tfs_part).reshape(1, 4, 4)
                    x_c = torch.matmul(inv_tfs, x_hat_homo)[:, :3, 0]
                    n_c = torch.matmul(inv_tfs[:, :3, :3], normal_points.reshape(-1, 3, 1))[:, :3,
                          0]
                    # x_c_valid_mask = torch.logical_not(torch.logical_or((x_c>bound).any(dim=-1), (x_c<-bound).any(dim=-1)))
                    occlusion = recon_occlusion_1spp(
                        H=points.shape[0],
                        W=1,
                        bound=bound,
                        points=x_c.clamp(min=-bound, max=bound).contiguous(),
                        normals=n_c.contiguous(),
                        occlusion_coefficients=occlusion_coefficients_list[i],
                        occlusion_ids=occlusion_ids_list[i],
                        aabb=aabb,
                        degree=occlusion_degree,
                        # sample_rays = 256,
                    )
                    # import pdb;pdb.set_trace()
                    occlusion = occlusion.reshape(points.shape[0])
                    part_occlusion_list.append(occlusion)
                    # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

                part_occlusion_list = torch.stack(part_occlusion_list, dim=-1) # [N, 9]
                occlusion = torch.where(
                    points_group_one_hot > 0.5,
                    self_occlusion_list,
                    part_occlusion_list
                )
                # combine_all_type_occlusion
                occlusion_per_gs = occlusion.prod(dim=-1) # [N]
                # gasussian splatting
                occlusion = occlusion_per_gs
                if rendering_type == 'forward_pbr':
                    occlusion = occlusion_per_gs
                    irradiance = torch.zeros_like(occlusion)  # [N, 1]

            else:
                occlusion = torch.ones_like(pc.get_opacity)  # [N, 1]
                irradiance = torch.zeros_like(pc.get_opacity)  # [N, 1]
            iter_end_occ.record()
            torch.cuda.synchronize()
            elapsed_occ = iter_start_occ.elapsed_time(iter_end_occ)
            times_occ.append(elapsed_occ)
            if rendering_type == 'forward_pbr':
                iter_start_shading.record()
                pbr_result_gs = pbr_shading_gs(deformed_gaussian=pc,
                                               camera_center=c2w[:3, 3],
                                               # TODO check if the camera center is correct c2w[:3, 3]
                                               light=scene.cubemap,
                                               occlusion=occlusion,  # [pc, 1]
                                               irradiance=irradiance,  # [pc, 1]
                                               brdf_lut=brdf_lut)
                iter_end_shading.record()
                torch.cuda.synchronize()
                elapsed_shading = iter_start_shading.elapsed_time(iter_end_shading)
                times_shading.append(elapsed_shading)

                iter_start_rasterization.record()
                override_color = pbr_result_gs
                # TODO check shape should be [3, H, W]
                render_output = render_2dgs(data, pc, config.pipeline,
                                            background, scaling_modifier=1.0,
                                            override_color=override_color)
                pbr_result = render_output['render'].clamp(min=0.0, max=1.0)

                render_rgb = linear_to_srgb(pbr_result)
                iter_end_rasterization.record()
                torch.cuda.synchronize()
                elapsed_rasterization = iter_start_rasterization.elapsed_time(iter_end_rasterization)
                times_rasterization.append(elapsed_rasterization)
            else:
                iter_start_rasterization.record()
                black_color = [0, 0, 0]
                black_background = torch.tensor(black_color, dtype=torch.float32, device="cuda")
                rough_metal_occ = torch.cat(
                    [pc.get_roughness.reshape(-1, 1), pc.get_metallic.reshape(-1, 1), occlusion.reshape(-1, 1)], dim=-1)
                rough_metal_occ_output = render_fast_2dgs(data, pc, config.pipeline,
                                                     black_background, scaling_modifier=1.0,
                                                     override_color=rough_metal_occ)
                # TODO here, all the image log via wandb is min-max normalized, but the image saved to disk is not. (data = vis_util.make_grid(data, normalize=True))
                roughness_map = torch.clamp(rough_metal_occ_output["render"][[0]], 0.0, 1.0)
                # import pdb;pdb.set_trace()
                metallic_map = torch.clamp(rough_metal_occ_output["render"][[1]], 0.0, 1.0)


                occlusion_img = torch.clamp(rough_metal_occ_output["render"][[2]], 0.0, 1.0)


                albedo_align_output = render_fast_2dgs(data, pc, config.pipeline,
                                                  background, scaling_modifier=1.0,
                                                  override_color=pc.get_albedo)
                input_albedo = albedo_align_output["render"]

                unsigned_normal = flip_normals(pc, c2w)

                normal_align_output_dict = render_fast_2dgs(data, pc, config.pipeline,
                                                       black_background, scaling_modifier=1.0,
                                                       override_color=unsigned_normal)
                # renormalize the normal map
                normal_align_output = normal_align_output_dict["render"]
                normal_align_output = normal_align_output / (normal_align_output.norm(dim=0, keepdim=True) + 1e-10)

                iter_end_rasterization.record()
                torch.cuda.synchronize()
                elapsed_rasterization = iter_start_rasterization.elapsed_time(iter_end_rasterization)
                times_rasterization.append(elapsed_rasterization)

                iter_start_shading.record()

                H, W = data.image_height, data.image_width
                view_dirs = -get_world_camera_rays_from_intrinsic(data).reshape(H, W, 3)

                alpha_mask = albedo_align_output["rend_alpha"].permute(1, 2, 0)
                pbr_result = pbr_shading(
                    light=scene.cubemap,
                    normals=normal_align_output.permute(1, 2, 0),  # [H, W, 3]
                    # normals=render_output["rend_normal"].permute(1, 2, 0).detach(),  # [H, W, 3]
                    view_dirs=view_dirs,
                    mask=alpha_mask,  # [H, W, 1]
                    albedo=input_albedo.permute(1, 2, 0),  # [H, W, 3]
                    roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                    metallic=metallic_map.permute(1, 2, 0),  # [H, W, 1]
                    tone=False,
                    gamma=True,
                    occlusion=occlusion_img.reshape(H, W, 1),
                    irradiance=torch.zeros_like(occlusion_img).reshape(H, W, 1),
                    brdf_lut=brdf_lut,
                    background=background,
                )
                iter_end_shading.record()
                torch.cuda.synchronize()
                elapsed_shading = iter_start_shading.elapsed_time(iter_end_shading)
                times_shading.append(elapsed_shading)



                # mask = rough_metal_occ_output["rend_alpha"].permute(1, 2, 0)
                # # normals = rough_metal_occ_output["rend_normal"].permute(1, 2, 0)
                # normals = normal_align_output.permute(1, 2, 0)
                # view_dirs = view_dirs
                # albedo = input_albedo.permute(1, 2, 0)
                # roughness = roughness_map.permute(1, 2, 0)
                # metallic = metallic_map.permute(1, 2, 0)
                # occlusion = occlusion_img.reshape(H, W, 1)
                # irradiance = torch.zeros_like(occlusion_img).reshape(H, W, 1)
                #
                # normal_mask = ~(normals == 0).all(dim=-1).reshape(H*W)
                # mask = mask.reshape(H*W)[normal_mask]
                # normals = normals.reshape(H*W, 3)[normal_mask]
                # view_dirs = view_dirs.reshape(H*W, 3)[normal_mask]
                # albedo = albedo.reshape(H*W, 3)[normal_mask]
                # roughness = roughness.reshape(H*W)[normal_mask]
                # metallic = metallic.reshape(H*W)[normal_mask]
                # occlusion = occlusion.reshape(H*W)[normal_mask]
                # irradiance = irradiance.reshape(H*W)[normal_mask]
                # # import pdb;pdb.set_trace()
                #
                #
                # pbr_result = pbr_shading(
                #     light=scene.cubemap,
                #     normals = normals.reshape(-1, 1, 3),
                #     # normals=normal_align_output.permute(1, 2, 0),  # [H, W, 3]
                #     # normals=render_output["rend_normal"].permute(1, 2, 0).detach(),  # [H, W, 3]
                #     view_dirs=view_dirs.reshape(-1, 1, 3),
                #     mask=mask.reshape(-1, 1, 1),  # [H, W, 1]
                #     albedo=albedo.reshape(-1, 1, 3),  # [H, W, 3]
                #     roughness=roughness.reshape(-1, 1, 1),  # [H, W, 1]
                #     metallic=metallic.reshape(-1, 1, 1),  # [H, W, 1]
                #     tone=False,
                #     gamma=True,
                #     occlusion=occlusion.reshape(-1, 1, 1),
                #     irradiance=irradiance.reshape(-1, 1, 1),
                #     brdf_lut=brdf_lut,
                #     background=background,
                #     return_background=False,
                # )
                # output_pbr = torch.ones(H*W, 3, device="cuda") * background.reshape(1, 3)
                # output_pbr[normal_mask] = pbr_result['render_rgb'].reshape(-1, 3)
                # render_rgb = output_pbr.reshape(H, W, 3)


            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)
            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        print(f"Average time: {_time:.2f} ms")
        _time_lbs = np.mean(times_lbs[1:])
        print(f"Average time lbs: {_time_lbs:.2f} ms")
        _time_occ = np.mean(times_occ[1:])
        print(f"Average time occ: {_time_occ:.2f} ms")
        _time_shading = np.mean(times_shading[1:])
        print(f"Average time shading: {_time_shading:.2f} ms")
        _time_rasterization = np.mean(times_rasterization[1:])
        print(f"Average time rasterization: {_time_rasterization:.2f} ms")
        _time_diffuse = np.mean(times_diffuse[1:])
        print(f"Average time diffuse: {_time_diffuse:.2f} ms")
        print(f"2dgs number: {scene.gaussians.get_xyz.shape[0]}")


@hydra.main(version_base=None, config_path="configs", config_name="config_peoplesnapshot")
def main(config_explicit_implicit):
    print(OmegaConf.to_yaml(config_explicit_implicit))
    config = config_explicit_implicit.explicit_branch

    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar-test',
        entity='jiangzeren',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        # predict(config)
        test(config)
    else:
        raise ValueError

if __name__ == "__main__":
    main()