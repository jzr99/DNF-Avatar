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
from gs_ir import recon_occlusion, IrradianceVolumes, recon_occlusion_1spp
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

def flip_normals(pc, c2w, threshold=-0.3):
    unsigned_normal = pc.get_normal.clone()
    camera_center = c2w[:3, 3]
    deformed_position = pc.get_xyz
    norm_view_dir = (camera_center - deformed_position) / (
                torch.norm(camera_center - deformed_position, dim=-1, keepdim=True) + 1e-8)
    unsigned_normal[(norm_view_dir * unsigned_normal).sum(dim=-1) < threshold] = -unsigned_normal[
        (norm_view_dir * unsigned_normal).sum(dim=-1) < threshold]

    return unsigned_normal


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

        # import pdb;pdb.set_trace()
        if config.dataset.name == 'people_snapshot' or config.dataset.name == 'animation':
            scene.gaussians.density_prune_distillation(scene.init_point_cloud,10,0.001)
        # scene.gaussians.prune_small_opacity()
        # scene.gaussians.prune_large_condition_number()
        print('shape after filter!!', scene.gaussians.get_xyz.shape)
        # import pdb; pdb.set_trace()

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

        normal_splat_path = os.path.join(config.exp_dir, config.suffix, 'normal_splat')
        makedirs(normal_splat_path, exist_ok=True)

        opacity_path = os.path.join(config.exp_dir, config.suffix, 'opacity')
        makedirs(opacity_path, exist_ok=True)

        occlusion_path = os.path.join(config.exp_dir, config.suffix, 'occlusion')
        makedirs(occlusion_path, exist_ok=True)

        render_pbr_path = os.path.join(config.exp_dir, config.suffix, 'renders_pbr')
        makedirs(render_pbr_path, exist_ok=True)

        render_deferred_pbr_path = os.path.join(config.exp_dir, config.suffix, 'renders_deferred_pbr')
        makedirs(render_deferred_pbr_path, exist_ok=True)

        render_deferred_pbr_bg_path = os.path.join(config.exp_dir, config.suffix, 'renders_deferred_pbr_bg')
        makedirs(render_deferred_pbr_bg_path, exist_ok=True)

        albedo_path = os.path.join(config.exp_dir, config.suffix, 'albedo')
        makedirs(albedo_path, exist_ok=True)

        metallic_path = os.path.join(config.exp_dir, config.suffix, 'metallic')
        makedirs(metallic_path, exist_ok=True)

        roughness_path = os.path.join(config.exp_dir, config.suffix, 'roughness')
        makedirs(roughness_path, exist_ok=True)

        background_path = os.path.join(config.exp_dir, config.suffix, 'background')
        makedirs(background_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        # evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()
        if config.dataset.name == 'people_snapshot' or config.dataset.name == 'animation':
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

        psnrs_pbr = []
        ssims_pbr = []
        lpipss_pbr = []

        psnrs_deferred_pbr = []
        ssims_deferred_pbr = []
        lpipss_deferred_pbr = []

        normal_error_list = []
        psnrs_pbr_albedo = []
        ssims_pbr_albedo = []
        lpipss_pbr_albedo = []

        occlusion_flag = True
        occlusion_ids_list =[]
        occlusion_coefficients_list =[]
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            # if idx!=52:
            #     continue
            view = scene.test_dataset[idx]
            data = view
            if data.data.get('hdri', None) is not None:
                # hdri = torch.from_numpy(hdri).cuda()
                res = scene.cubemap.base.data.shape[1]
                scene.cubemap.base.data = latlong_to_cubemap(data.hdri, [res, res])

            iter_start.record()

            # render_pkg = render(view, config.opt.iterations, scene, config.pipeline, background,
            #                     compute_loss=False, return_opacity=True, return_depth=True, return_normal=True, inference=True,pad_normal=False,derive_normal=True,)
            pc, loss_reg, colors_precomp = scene.convert_gaussians(data, config.opt.iterations, compute_loss=False)
            render_output = render_2dgs(data, pc, config.pipeline,
                                        background, scaling_modifier=1.0,
                                        override_color=colors_precomp)

            # import pdb;pdb.set_trace()

            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            black_color = [0, 0, 0]
            black_background = torch.tensor(black_color, dtype=torch.float32, device="cuda")
            albedo_output = render_2dgs(data, pc, config.pipeline,
                                        black_background, scaling_modifier=1.0,
                                        override_color=pc.get_albedo)
            # opacity_mask = albedo_output["rend_alpha"] > 0.95

            # TODO align the albedo map with ground truth
            pred_aligned_albedo = None
            three_channel_ratio = None
            if data.data.get('gt_albedo', None) is not None:
                H, W = albedo_output["render"].shape[1], albedo_output["render"].shape[2]
                pred_albedo = albedo_output["render"].clone().permute(1, 2, 0).reshape(-1, 3) # [HW, 3]
                gt_mask = data.original_mask.permute(1, 2, 0).reshape(-1) > 0.5 # [HW,]
                gt_albedo = data.gt_albedo.permute(1, 2, 0).reshape(-1, 3) # [HW, 3]
                three_channel_ratio = compute_albedo_rescale_factor(
                    gt_albedo, pred_albedo, gt_mask
                )
                # import pdb;pdb.set_trace()
                # # TODO check the mask shape
                # pred_mask = opacity_mask.reshape(-1)
                three_aligned_albedo = torch.zeros_like(gt_albedo)
                pred_mask = gt_mask
                three_aligned_albedo[pred_mask] = three_channel_ratio.reshape(1, 3) * pred_albedo[pred_mask]
                pred_aligned_albedo = three_aligned_albedo.reshape(H, W, 3).permute(2, 0, 1).clamp(min=0.0, max=1.0) # (3, H, W)
                # here if you do this, it means you are using the gt mask to align the albedo map
                # three_aligned_albedo[gt_mask] = (
                #         three_channel_ratio * pred_albedo[gt_mask]
                # ).clamp(min=0.0, max=1.0)

            rendering = render_output["render"]

            gt = view.original_image[:3, :, :]

            examples = []

            # examples = [wandb.Image(rendering[None], caption='render_{}'.format(view.image_name)),
            #              wandb.Image(gt[None], caption='gt_{}'.format(view.image_name))]

            # wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{view.image_name}.png"))
            torchvision.utils.save_image(render_output["rend_alpha"], os.path.join(opacity_path, f"opacity_{view.image_name}.png"))
            # torchvision.utils.save_image(render_pkg["depth_render"], os.path.join(depth_path, f"depth_{view.image_name}.png"))
            # torchvision.utils.save_image(render_pkg["normal_render"], os.path.join(normal_path, f"normal_{view.image_name}.png"))
            image = torch.clamp(render_output["render"], 0.0, 1.0)
            gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
            opacity_image = torch.clamp(render_output["rend_alpha"], 0.0, 1.0)

            wandb_img = wandb.Image(opacity_image[None],
                                    caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(image[None],
                                    caption=config['name'] + "_view_{}/render".format(data.image_name))
            examples.append(wandb_img)
            wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                data.image_name))
            examples.append(wandb_img)

            # w2c = data.world_view_transform.T  # [4, 4]
            w2c = data.w2c_opencv
            # import pdb;pdb.set_trace()
            normal_map = torch.clamp(transform_normals(w2c, render_output["rend_normal"]) * 0.5 + 0.5, 0.0, 1.0)
            wandb_img = wandb.Image(normal_map[None],
                                    caption=config['name'] + "_view_{}/normal_map".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(normal_map, os.path.join(normal_path, f"normal_{view.image_name}.png"))

            depth_map = torch.clamp(
                torch.from_numpy(
                    turbo_cmap(render_output["surf_depth"].cpu().numpy().squeeze())
                )
                .to(image.device)
                .permute(2, 0, 1), 0.0, 1.0)
            wandb_img = wandb.Image(depth_map[None],
                                    caption=config['name'] + "_view_{}/depth_map".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(depth_map, os.path.join(depth_path, f"depth_{view.image_name}.png"))

            albedo_map = torch.clamp(albedo_output["render"], 0.0, 1.0)
            wandb_img = wandb.Image(albedo_map[None],
                                    caption=config['name'] + "_view_{}/albedo_map".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(albedo_map, os.path.join(albedo_path, f"albedo_{view.image_name}.png"))

            if pred_aligned_albedo is not None:
                pred_aligned_albedo = torch.clamp(pred_aligned_albedo, 0.0, 1.0)
                wandb_img = wandb.Image(pred_aligned_albedo[None],
                                        caption=config['name'] + "_view_{}/albedo_map_aligned".format(data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(pred_aligned_albedo, os.path.join(albedo_path, f"albedo_aligned_{view.image_name}.png"))

            # import pdb;pdb.set_trace()

            normal_map_from_depth = torch.clamp(transform_normals(w2c, render_output["surf_normal"]) * 0.5 + 0.5, 0.0, 1.0)
            wandb_img = wandb.Image(normal_map_from_depth[None],
                                    caption=config['name'] + "_view_{}/normal_map_from_depth".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(normal_map_from_depth, os.path.join(normal_depth_path, f"normal_depth_{view.image_name}.png"))


            # metrics_test = evaluator(image, gt_image)
            # psnr_test += metrics_test["psnr"]
            # ssim_test += metrics_test["ssim"]
            # lpips_test += metrics_test["lpips"]

            # # pbr rendering
            # normal_map = render_pkg["normal_map"]  # [3, H, W]
            # albedo_map = render_pkg["albedo_map"]  # [3, H, W]
            # roughness_map = render_pkg["roughness_map"]  # [1, H, W]
            # metallic_map = render_pkg["metallic_map"]  # [1, H, W]

            # # formulate roughness
            # rmax, rmin = 1.0, 0.04
            # roughness_map = roughness_map * (rmax - rmin) + rmin

            # c2w = torch.inverse(data.world_view_transform.T)  # [4, 4]
            c2w = torch.inverse(data.w2c_opencv)  # [4, 4]
            # canonical_rays = get_canonical_rays_from_intrinsic(data)

            # NOTE: mask normal map by view direction to avoid skip value
            H, W = data.image_height, data.image_width
            # view_dirs = -(
            #     (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            #     .sum(dim=-1)
            #     .reshape(H, W, 3)
            # )  # [H, W, 3]
            view_dirs = -get_world_camera_rays_from_intrinsic(data).reshape(H, W, 3)

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
            if config.enable_occ_type == 'gaussian':
                # print("begin to use gaussian splatting to render occlusion")
                # deformed normal
                # normal_points = pc.get_normal
                normal_points = flip_normals(pc, c2w)
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
                    if config.enable_occ_1spp == False:
                        occlusion = recon_occlusion(
                            H=points.shape[0],
                            W=1,
                            bound=bound,
                            points=x_c.clamp(min=-bound, max=bound).contiguous(),
                            normals=n_c.contiguous(),
                            occlusion_coefficients=occlusion_coefficients_list[i],
                            occlusion_ids=occlusion_ids_list[i],
                            aabb=aabb,
                            degree=occlusion_degree,
                        )
                    else:
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
                        )
                    occlusion = occlusion.reshape(points.shape[0])
                    part_occlusion_list.append(occlusion)
                    # uncomment to visualize the occlusion map
                    # part_occ_output = render_2dgs(data, pc, config.pipeline,
                    #                                      black_background, scaling_modifier=1.0,
                    #                                      override_color=occlusion.reshape(-1, 1).repeat(1, 3))
                    # part_occ_map = torch.clamp(part_occ_output["render"][[0]], 0.0, 1.0)
                    # wandb_img = wandb.Image(part_occ_map[None],
                    #                         caption=config['name'] + "_view_{}_part{}/part_occ_map".format(data.image_name, i))
                    # examples.append(wandb_img)
                    # torchvision.utils.save_image(part_occ_map,
                    #                              os.path.join(occlusion_path, f"part_occ_{view.image_name}_part{i}.png"))

                    # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

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
                self_occlusion_list = []
                for i in range(9):
                    if config.enable_occ_1spp == False:
                        occlusion = recon_occlusion(
                            H=points.shape[0],
                            W=1,
                            bound=bound,
                            points=points.clamp(min=-bound, max=bound).contiguous(),
                            normals=normal_points.contiguous(),
                            occlusion_coefficients=occlusion_coefficients_list[i],
                            occlusion_ids=occlusion_ids_list[i],
                            aabb=aabb,
                            degree=occlusion_degree,
                            shift_ratio=0.1,
                        )
                    else:
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
                            shift_ratio=0.1,
                        )
                    occlusion = occlusion.reshape(points.shape[0])
                    self_occlusion_list.append(occlusion)
                    # uncomment to visualize the occlusion map
                    # self_occ_output = render_2dgs(data, pc, config.pipeline,
                    #                               black_background, scaling_modifier=1.0,
                    #                               override_color=occlusion.reshape(-1, 1).repeat(1, 3))
                    # self_occ_map = torch.clamp(self_occ_output["render"][[0]], 0.0, 1.0)
                    # wandb_img = wandb.Image(self_occ_map[None],
                    #                         caption=config['name'] + "_view_{}_self{}/self_occ_map".format(
                    #                             data.image_name, i))
                    # examples.append(wandb_img)
                    # torchvision.utils.save_image(self_occ_map,
                    #                              os.path.join(occlusion_path,
                    #                                           f"self_occ_{view.image_name}_part{i}.png"))

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
                for idx, part in enumerate(part_group):
                    # mask different body part
                    _, points_weights_smpl = scene.converter.deformer.rigid.query_weights(points)
                    points_weights_part_idx = points_weights_smpl.argmax(dim=1)

                    # select point based on the part group
                    part = torch.tensor(part).cuda()
                    point_mask = points_weights_part_idx.reshape(points_weights_part_idx.shape[0], 1) == part.reshape(1, -1)
                    point_mask = point_mask.any(dim=1)
                    points_group_one_hot[point_mask, idx] = 1

                part_occlusion_list = torch.stack(part_occlusion_list, dim=-1) # [N, 9]
                self_occlusion_list = torch.stack(self_occlusion_list, dim=-1) # [N, 9]
                occlusion = torch.where(
                    points_group_one_hot > 0.5,
                    self_occlusion_list,
                    part_occlusion_list
                )
                # import pdb; pdb.set_trace()
                # combine_all_type_occlusion
                occlusion_per_gs = occlusion.prod(dim=-1) # [N]
                # gasussian splatting
                # if rendering_type == 'forward_pbr':
                occlusion = occlusion_per_gs
                irradiance = torch.zeros_like(occlusion)  # [N, 1]


                # for visualization
                occlusion_self_vis = torch.where(
                    points_group_one_hot > 0.5,
                    self_occlusion_list,
                    torch.ones_like(self_occlusion_list, device=self_occlusion_list.device)
                )
                occlusion_self_vis = occlusion_self_vis.prod(dim=-1).reshape(-1, 1)  # [N]

                self_occ_output = render_2dgs(data, pc, config.pipeline,
                                              black_background, scaling_modifier=1.0,
                                              override_color=occlusion_self_vis.reshape(-1, 1).repeat(1, 3))
                self_occ_map = torch.clamp(self_occ_output["render"][[0]], 0.0, 1.0)
                wandb_img = wandb.Image(self_occ_map[None],
                                        caption=config['name'] + "_view_{}_self_all/self_occ_map".format(
                                            data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(self_occ_map,
                                             os.path.join(occlusion_path,
                                                          f"self_occ_{view.image_name}_all.png"))



                # occlusion_img = occlusion.clamp(min=0.0, max=1.0).permute(2, 0, 1)
                # torchvision.utils.save_image(occlusion_img,
                #                              os.path.join(occlusion_path, f"occlusion_{view.image_name}.png"))
                # wandb_img = wandb.Image(occlusion_img[None],
                #                         caption=config['name'] + "_view_{}/occlusion".format(data.image_name))
                # examples.append(wandb_img)

            else:
                # occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                # irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                occlusion = torch.ones_like(pc.get_opacity)  # [N, 1]
                irradiance = torch.zeros_like(pc.get_opacity)  # [N, 1]

            black_color =  [0, 0, 0]
            black_background = torch.tensor(black_color, dtype=torch.float32, device="cuda")
            rough_metal_occ = torch.cat([pc.get_roughness.reshape(-1, 1), pc.get_metallic.reshape(-1, 1), occlusion.reshape(-1, 1)], dim=-1)
            rough_metal_occ_output = render_2dgs(data, pc, config.pipeline,
                                        black_background, scaling_modifier=1.0,
                                        override_color=rough_metal_occ)
            # TODO here, all the image log via wandb is min-max normalized, but the image saved to disk is not. (data = vis_util.make_grid(data, normalize=True))
            roughness_map = torch.clamp(rough_metal_occ_output["render"][[0]], 0.0, 1.0)
            wandb_img = wandb.Image(roughness_map[None],
                                    caption=config['name'] + "_view_{}/roughness_map".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(roughness_map,
                                         os.path.join(roughness_path, f"roughness_{view.image_name}.png"))
            # import pdb;pdb.set_trace()
            metallic_map = torch.clamp(rough_metal_occ_output["render"][[1]], 0.0, 1.0)
            wandb_img = wandb.Image(metallic_map[None],
                                    caption=config['name'] + "_view_{}/metallic_map".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(metallic_map,
                                         os.path.join(metallic_path, f"metallic_{view.image_name}.png"))

            occlusion_img = torch.clamp(rough_metal_occ_output["render"][[2]], 0.0, 1.0)
            wandb_img = wandb.Image(occlusion_img[None],
                                    caption=config['name'] + "_view_{}/occlusion".format(data.image_name))
            examples.append(wandb_img)
            torchvision.utils.save_image(occlusion_img,
                                         os.path.join(occlusion_path, f"occlusion_{view.image_name}.png"))

            # normal_mask = render_pkg["normal_mask"]  # [1, H, W]
            scene.cubemap.build_mips()  # build mip for environment light
            # TODO we should chekc metallic_map is valid or not
            metallic = True # default is false
            # forward_pbr
            if True:
                print(three_channel_ratio)
                pbr_result_gs = pbr_shading_gs(deformed_gaussian=pc,
                                               camera_center=c2w[:3, 3],
                                               # TODO check if the camera center is correct c2w[:3, 3]
                                               light=scene.cubemap,
                                               occlusion=occlusion,  # [pc, 1]
                                               irradiance=irradiance,  # [pc, 1]
                                               brdf_lut=brdf_lut,
                                               three_channel_ratio=three_channel_ratio)
                override_color = pbr_result_gs
                # TODO check shape should be [3, H, W]
                render_output = render_2dgs(data, pc, config.pipeline,
                                            background, scaling_modifier=1.0,
                                            override_color=override_color)
                pbr_result = render_output['render'].clamp(min=0.0, max=1.0)

                render_rgb = linear_to_srgb(pbr_result)
                # TODO check whether it is needed
                # render_rgb = torch.where(
                #     normal_mask,
                #     render_rgb,
                #     background[:, None, None],
                # )
                # TODO update weights
                # pbr_render_loss = l1_loss(render_rgb, gt_image)
                # loss += pbr_render_loss
                wandb_img = wandb.Image(render_rgb[None],
                                        caption=config['name'] + "_view_{}/pbr_image".format(data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(render_rgb,
                                             os.path.join(render_pbr_path, f"render_pbr_{view.image_name}.png"))

            if True:
                # input_albedo = pred_aligned_albedo if pred_aligned_albedo is not None else albedo_output["render"]
                albedo_ratio = three_channel_ratio if three_channel_ratio is not None else torch.ones(3, device=pc.get_xyz.device)
                albedo_align_output = render_2dgs(data, pc, config.pipeline,
                                            background, scaling_modifier=1.0,
                                            override_color=pc.get_albedo * albedo_ratio.reshape(1, 3))
                input_albedo = albedo_align_output["render"]

                unsigned_normal = flip_normals(pc, c2w)
                # unsigned_normal = pc.get_normal

                normal_align_output_dict = render_2dgs(data, pc, config.pipeline,
                                                  black_background, scaling_modifier=1.0,
                                                  override_color=unsigned_normal)
                # renormalize the normal map
                normal_align_output = normal_align_output_dict["render"]

                # normal_align_output[(view_dirs * normal_align_output.permute(1, 2, 0)).sum(dim=-1) < 0] = -normal_align_output[
                #     (view_dirs * normal_align_output.permute(1, 2, 0)).sum(dim=-1) < 0]

                normal_align_map = torch.clamp(transform_normals(w2c, normal_align_output) * 0.5 + 0.5, 0.0, 1.0)
                wandb_img = wandb.Image(normal_align_map[None],
                                        caption=config['name'] + "_view_{}/normal_align_map".format(data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(normal_align_map, os.path.join(normal_splat_path, f"normal_{view.image_name}.png"))

                normal_align_output = normal_align_output / (normal_align_output.norm(dim=0, keepdim=True) + 1e-10)

                alpha_mask = render_output["rend_alpha"].permute(1, 2, 0)
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
                # render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
                # # TODO check whether it is needed
                # render_rgb = torch.where(
                #     normal_mask,
                #     render_rgb,
                #     self.background[:, None, None],
                # )
                diffuse_rgb = pbr_result["diffuse_rgb"] * alpha_mask + (1 - alpha_mask) * background[None, None, :]
                specular_rgb = pbr_result["specular_rgb"] * alpha_mask + (1 - alpha_mask) * background[None, None, :]

                diffuse_rgb = (
                    diffuse_rgb.clamp(min=0.0, max=1.0).permute(2, 0, 1)
                )  # [3, H, W]
                specular_rgb = (
                    specular_rgb.clamp(min=0.0, max=1.0).permute(2, 0, 1)
                )  # [3, H, W]
                render_deferred_rgb = (
                    pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                )  # [3, H, W]
                render_rgb_background = (
                    pbr_result["render_rgb_background"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                )
                background_light = (
                    pbr_result["background_light"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                )
                # # NOTE: mask render_rgb by depth map
                # background = renderArgs[1]
                # render_rgb = torch.where(
                #     # render_pkg["opacity_mask"],
                #     normal_mask,
                #     render_rgb,
                #     background[:, None, None],
                # )
                # diffuse_rgb = torch.where(
                #     # render_pkg["opacity_mask"],
                #     normal_mask,
                #     diffuse_rgb,
                #     background[:, None, None],
                # )
                # specular_rgb = torch.where(
                #     # render_pkg["opacity_mask"],
                #     normal_mask,
                #     specular_rgb,
                #     background[:, None, None],
                # )
                pbr_image = torch.cat(
                    [render_deferred_rgb, diffuse_rgb, specular_rgb], dim=2
                )  # [3, H, 3W]

                torchvision.utils.save_image(background_light,
                                             os.path.join(background_path,
                                                          f"background.png"))

                wandb_img = wandb.Image(pbr_image[None],
                                        caption=config['name'] + "_view_{}/pbr_deferred_image".format(data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(pbr_image,
                                                os.path.join(render_deferred_pbr_path, f"render_pbr_{view.image_name}.png"))

                wandb_img = wandb.Image(render_rgb_background[None],
                                        caption=config['name'] + "_view_{}/pbr_deferred_image_bg".format(data.image_name))
                examples.append(wandb_img)
                torchvision.utils.save_image(render_rgb_background,
                                             os.path.join(render_deferred_pbr_bg_path,
                                                          f"render_pbr_bg_{view.image_name}.png"))


            wandb.log({"test_images": examples})
            examples.clear()
            #
            # l1_test_pbr += l1_loss(render_rgb, gt_image).mean().double()
            # metrics_test_pbr = evaluator(render_rgb, gt_image)
            # psnr_test_pbr += metrics_test_pbr["psnr"]
            # ssim_test_pbr += metrics_test_pbr["ssim"]
            # lpips_test_pbr += metrics_test_pbr["lpips"]

            # evaluate
            if config.evaluate:
                metrics = evaluator(
                    rendering,
                    gt,
                    valid_mask=data.valid_msk,
                )
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])

                metrics_pbr = evaluator(
                    render_rgb,
                    gt,
                    valid_mask=data.valid_msk,
                )
                psnrs_pbr.append(metrics_pbr['psnr'])
                ssims_pbr.append(metrics_pbr['ssim'])
                lpipss_pbr.append(metrics_pbr['lpips'])

                metrics_deferred_pbr = evaluator(
                    render_deferred_rgb,
                    gt,
                    valid_mask=data.valid_msk,
                )
                psnrs_deferred_pbr.append(metrics_deferred_pbr['psnr'])
                ssims_deferred_pbr.append(metrics_deferred_pbr['ssim'])
                lpipss_deferred_pbr.append(metrics_deferred_pbr['lpips'])

                camera_rend_normal = transform_normals(w2c, normal_align_output_dict["render"]).permute(1, 2, 0).reshape(-1, 3)
                # camera_rend_normal = transform_normals(w2c, render_output["rend_normal"]).permute(1, 2, 0).reshape(-1, 3)
                # import pdb;pdb.set_trace()
                # check gt_normal data.gt_albedo shape
                if data.gt_normal is not None:
                    normal_error = evaluator.evaluate_normal(camera_rend_normal, data.gt_normal.permute(1, 2, 0).reshape(-1, 3), data.original_mask.reshape(-1))
                    normal_error_list.append(normal_error)
                else:
                    normal_error_list.append(torch.tensor([0.], device='cuda'))

                if pred_aligned_albedo is not None:
                    metrics_pbr_albedo = evaluator(
                        pred_aligned_albedo,
                        data.gt_albedo,
                        valid_mask=gt_mask.reshape(H, W),
                    )
                    psnrs_pbr_albedo.append(metrics_pbr_albedo['psnr'])
                    ssims_pbr_albedo.append(metrics_pbr_albedo['ssim'])
                    lpipss_pbr_albedo.append(metrics_pbr_albedo['lpips'])
                else:
                    psnrs_pbr_albedo.append(torch.tensor([0.], device='cuda'))
                    ssims_pbr_albedo.append(torch.tensor([0.], device='cuda'))
                    lpipss_pbr_albedo.append(torch.tensor([0.], device='cuda'))

            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
                psnrs_pbr.append(torch.tensor([0.], device='cuda'))
                ssims_pbr.append(torch.tensor([0.], device='cuda'))
                lpipss_pbr.append(torch.tensor([0.], device='cuda'))
                psnrs_deferred_pbr.append(torch.tensor([0.], device='cuda'))
                ssims_deferred_pbr.append(torch.tensor([0.], device='cuda'))
                lpipss_deferred_pbr.append(torch.tensor([0.], device='cuda'))
                normal_error_list.append(torch.tensor([0.], device='cuda'))
                psnrs_pbr_albedo.append(torch.tensor([0.], device='cuda'))
                ssims_pbr_albedo.append(torch.tensor([0.], device='cuda'))
                lpipss_pbr_albedo.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _psnr_pbr = torch.mean(torch.stack(psnrs_pbr))
        _ssim_pbr = torch.mean(torch.stack(ssims_pbr))
        _lpips_pbr = torch.mean(torch.stack(lpipss_pbr))
        _psnr_deferred_pbr = torch.mean(torch.stack(psnrs_deferred_pbr))
        _ssim_deferred_pbr = torch.mean(torch.stack(ssims_deferred_pbr))
        _lpips_deferred_pbr = torch.mean(torch.stack(lpipss_deferred_pbr))
        _normal_error = torch.mean(torch.stack(normal_error_list))
        _psnr_pbr_albedo = torch.mean(torch.stack(psnrs_pbr_albedo))
        _ssim_pbr_albedo = torch.mean(torch.stack(ssims_pbr_albedo))
        _lpips_pbr_albedo = torch.mean(torch.stack(lpipss_pbr_albedo))
        _time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time,
                   'metrics/psnr_pbr': _psnr_pbr,
                   'metrics/ssim_pbr': _ssim_pbr,
                   'metrics/lpips_pbr': _lpips_pbr,
                   'metrics/psnr_deferred_pbr': _psnr_deferred_pbr,
                   'metrics/ssim_deferred_pbr': _ssim_deferred_pbr,
                   'metrics/lpips_deferred_pbr': _lpips_deferred_pbr,
                   'metrics/normal_error': _normal_error,
                   'metrics/psnr_pbr_albedo': _psnr_pbr_albedo,
                   'metrics/ssim_pbr_albedo': _ssim_pbr_albedo,
                   'metrics/lpips_pbr_albedo': _lpips_pbr_albedo,})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 psnr_pbr=_psnr_pbr.cpu().numpy(),
                 ssim_pbr=_ssim_pbr.cpu().numpy(),
                 lpips_pbr=_lpips_pbr.cpu().numpy(),
                 psnr_deferred_pbr=_psnr_deferred_pbr.cpu().numpy(),
                 ssim_deferred_pbr=_ssim_deferred_pbr.cpu().numpy(),
                 lpips_deferred_pbr=_lpips_deferred_pbr.cpu().numpy(),
                 normal_error=_normal_error.cpu().numpy(),
                 psnr_pbr_albedo=_psnr_pbr_albedo.cpu().numpy(),
                 ssim_pbr_albedo=_ssim_pbr_albedo.cpu().numpy(),
                 lpips_pbr_albedo=_lpips_pbr_albedo.cpu().numpy(),
                 time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
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