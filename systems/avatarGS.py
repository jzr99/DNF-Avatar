import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import render
from gaussian_renderer_pbr import render, render_fast
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator, RANAEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss
from omegaconf import OmegaConf
import wandb
import lpips
import numpy as np
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union
import nvdiffrast.torch as dr
import kornia
from pbr import get_brdf_lut, pbr_shading, pbr_shading_gs, linear_to_srgb
# from utils.camera_utils import get_canonical_rays,
from utils.camera_utils import get_world_camera_rays_from_intrinsic
from utils.image_utils import turbo_cmap
from gs_ir import recon_occlusion, IrradianceVolumes
from utils.general_utils import use_plyfile
from utils.general_utils import build_rotation


class AvatarGS:
    def __init__(self, config):
        super().__init__()

        OmegaConf.set_struct(config, False)  # allow adding new values to config
        config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
        os.makedirs(config.exp_dir, exist_ok=True)
        config.checkpoint_iterations.append(config.opt.iterations)

        self.GS_config = config
        self.model = config.model
        self.dataset = config.dataset
        self.opt = config.opt
        self.pipe = config.pipeline
        self.testing_iterations = config.test_iterations
        self.testing_interval = config.test_interval
        self.saving_iterations = config.save_iterations
        self.checkpoint_iterations = config.checkpoint_iterations
        # TODO: we should support checkpoint loading for the optimizer
        checkpoint = config.start_checkpoint
        self.debug_from = config.debug_from

        # define lpips
        lpips_type = config.opt.get('lpips_type', 'vgg')
        self.loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()  # for training
        if self.dataset.name == 'people_snapshot':
            self.evaluator = PSEvaluator()
        elif self.dataset.name == 'rana':
            self.evaluator = RANAEvaluator()
        else:
            self.evaluator = Evaluator()
            raise NotImplementedError(f"Unknown dataset {self.dataset.name}")

        self.gaussians = GaussianModel(self.model.gaussian)
        self.scene = Scene(config, self.gaussians, config.exp_dir)
        self.scene.train()

        self.gaussians.training_setup(self.opt)
        # if checkpoint:
        #     scene.load_checkpoint(checkpoint)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.iter = 0
        self.data_stack = list(range(len(self.scene.train_dataset)))
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm(range(0, self.opt.iterations), desc="Training progress")

        # prepare for PBR
        self.brdf_lut = get_brdf_lut().cuda()
        self.envmap_dirs = self.get_envmap_dirs()
        self.occlusion_flag=True
        self.occlusion_ids_list = []
        self.occlusion_coefficients_list = []

        self.pbr_iteration = config.model.pbr_iteration

        self.rendering_type = config.rendering_type # 'forward_pbr' or 'diff_pbr'

        # self.distill_adapt_metallic_layer = nn.Conv1d(1, 1, 1, stride=1, device='cuda')
        # self.distill_adapt_roughness_layer = nn.Conv1d(1, 1, 1, stride=1, device='cuda')

        # canonical_rays = scene.get_canonical_rays()

    def transform_normals(self, w2c, normals):
        """ Convert world-space normal map into OpenGL camera space
        """
        H, W = normals.shape[1], normals.shape[2]
        normals = normals.reshape(3, H * W).transpose(1, 0)
        # Convert to camera space, if necessary
        normals = torch.matmul(normals, w2c[:3, :3].transpose(0, 1))

        # Convert OpenCV to OpenGL convention
        normals = normals * torch.tensor([1.0, -1.0, -1.0], device=normals.device)
        normals = normals.transpose(1, 0).reshape(3, H, W)

        return normals

    def get_tv_loss(
            self,
            gt_image: torch.Tensor,  # [3, H, W]
            prediction: torch.Tensor,  # [C, H, W]
            pad: int = 1,
            step: int = 1,
    ) -> torch.Tensor:
        if pad > 1:
            gt_image = F.avg_pool2d(gt_image, pad, pad)
            prediction = F.avg_pool2d(prediction, pad, pad)
        rgb_grad_h = torch.exp(
            -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        rgb_grad_w = torch.exp(
            -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
        tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
        tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

        if step > 1:
            for s in range(2, step + 1):
                rgb_grad_h = torch.exp(
                    -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
                )  # [1, H-1, W]
                rgb_grad_w = torch.exp(
                    -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
                )  # [1, H-1, W]
                tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
                tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
                tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

        return tv_loss

    def get_masked_tv_loss(
            self,
            mask: torch.Tensor,  # [1, H, W]
            gt_image: torch.Tensor,  # [3, H, W]
            prediction: torch.Tensor,  # [C, H, W]
            erosion: bool = False,
    ) -> torch.Tensor:
        rgb_grad_h = torch.exp(
            -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        rgb_grad_w = torch.exp(
            -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
        )  # [1, H-1, W]
        tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
        tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

        # erode mask
        mask = mask.float()
        if erosion:
            kernel = mask.new_ones([7, 7])
            mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
        mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
        mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

        tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

        return tv_loss

    def get_envmap_dirs(self, res: List[int] = [512, 1024]) -> torch.Tensor:
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )

        sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
        sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

        reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
        return reflvec


    def forward(self):
        pass

    def GS_C(self, iteration, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = OmegaConf.to_container(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            value_list = [0] + value
            i = 0
            current_step = iteration
            while i < len(value_list):
                if current_step >= value_list[i]:
                    i += 2
                else:
                    break
            value = value_list[i - 1]
        return value

    def validation(self, iteration, testing_iterations, testing_interval, scene: Scene, evaluator, renderArgs):
        # Report test and samples of training set
        if testing_interval > 0:
            if not iteration % testing_interval == 0:
                return
        else:
            if not iteration in testing_iterations:
                return

        scene.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},
                              {'name': 'train', 'cameras': [idx for idx in range(0, len(scene.train_dataset),
                                                                                 len(scene.train_dataset) // 10)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                examples = []
                for idx, data_idx in enumerate(config['cameras']):
                    data = getattr(scene, config['name'] + '_dataset')[data_idx]
                    render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                    opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

                    wandb_img = wandb.Image(opacity_image[None],
                                            caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(image[None],
                                            caption=config['name'] + "_view_{}/render".format(data.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                        data.image_name))
                    examples.append(wandb_img)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    metrics_test = evaluator(image, gt_image)
                    psnr_test += metrics_test["psnr"]
                    ssim_test += metrics_test["ssim"]
                    lpips_test += metrics_test["lpips"]

                    wandb.log({config['name'] + "_images": examples})
                    examples.clear()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                wandb.log({
                    config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                    config['name'] + '/loss_viewpoint - psnr': psnr_test,
                    config['name'] + '/loss_viewpoint - ssim': ssim_test,
                    config['name'] + '/loss_viewpoint - lpips': lpips_test,
                })

        wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
        wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
        torch.cuda.empty_cache()
        scene.train()


    def validation_pbr(self, iteration, testing_iterations, testing_interval, scene: Scene, evaluator, renderArgs):
        # Report test and samples of training set
        if testing_interval > 0:
            if not iteration % testing_interval == 0:
                return
        else:
            if not iteration in testing_iterations:
                return

        scene.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': list(range(len(scene.test_dataset)))},
                              {'name': 'train', 'cameras': [idx for idx in range(0, len(scene.train_dataset),
                                                                                 len(scene.train_dataset) // 10)]})

        # distill env_map
        envmap = dr.texture(
            self.scene.cubemap.base[None, ...],
            self.envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        envmap = envmap.clamp(min=0.0, max=1.0).permute(2, 0, 1) # [3, H, W]
        wandb_img = wandb.Image(envmap[None],
                                caption= "env_map/env_map")
        wandb.log({"train_envmap": [wandb_img]})


        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                l1_test_pbr = 0.0
                psnr_test_pbr = 0.0
                ssim_test_pbr = 0.0
                lpips_test_pbr = 0.0
                examples = []
                for idx, data_idx in enumerate(config['cameras']):
                    data = getattr(scene, config['name'] + '_dataset')[data_idx]
                    render_pkg = render(data, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True, inference=True, derive_normal=True,)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                    opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)

                    wandb_img = wandb.Image(opacity_image[None],
                                            caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(image[None],
                                            caption=config['name'] + "_view_{}/render".format(data.image_name))
                    examples.append(wandb_img)
                    wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                        data.image_name))
                    examples.append(wandb_img)

                    # w2c = data.world_view_transform.T
                    w2c = data.w2c_opencv
                    normal_map = torch.clamp(self.transform_normals(w2c, render_pkg["normal_map"]) * 0.5 + 0.5, 0.0, 1.0)
                    wandb_img = wandb.Image(normal_map[None],
                                            caption=config['name'] + "_view_{}/normal_map".format(data.image_name))
                    examples.append(wandb_img)

                    depth_map = torch.clamp(
                        torch.from_numpy(
                            turbo_cmap(render_pkg["depth_map"].cpu().numpy().squeeze())
                        )
                        .to(image.device)
                        .permute(2, 0, 1), 0.0, 1.0)
                    wandb_img = wandb.Image(depth_map[None],
                                            caption=config['name'] + "_view_{}/depth_map".format(data.image_name))
                    examples.append(wandb_img)

                    albedo_map = torch.clamp(render_pkg["albedo_map"], 0.0, 1.0)
                    wandb_img = wandb.Image(albedo_map[None],
                                            caption=config['name'] + "_view_{}/albedo_map".format(data.image_name))
                    examples.append(wandb_img)

                    roughness_map = torch.clamp(render_pkg["roughness_map"], 0.0, 1.0)
                    wandb_img = wandb.Image(roughness_map[None],
                                            caption=config['name'] + "_view_{}/roughness_map".format(data.image_name))
                    examples.append(wandb_img)

                    metallic_map = torch.clamp(render_pkg["metallic_map"], 0.0, 1.0)
                    wandb_img = wandb.Image(metallic_map[None],
                                            caption=config['name'] + "_view_{}/metallic_map".format(data.image_name))
                    examples.append(wandb_img)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    metrics_test = evaluator(
                        image,
                        gt_image,
                        valid_mask=data.valid_msk,
                    )
                    psnr_test += metrics_test["psnr"]
                    ssim_test += metrics_test["ssim"]
                    lpips_test += metrics_test["lpips"]

                    # pbr rendering
                    normal_map = render_pkg["normal_map"] # [3, H, W]
                    albedo_map = render_pkg["albedo_map"]  # [3, H, W]
                    roughness_map = render_pkg["roughness_map"]  # [1, H, W]
                    metallic_map = render_pkg["metallic_map"]  # [1, H, W]

                    # # formulate roughness
                    # rmax, rmin = 1.0, 0.04
                    # roughness_map = roughness_map * (rmax - rmin) + rmin

                    # c2w = torch.inverse(data.world_view_transform.T)  # [4, 4]
                    c2w = torch.inverse(data.w2c_opencv)
                    # canonical_rays = get_canonical_rays_from_intrinsic(data)

                    # NOTE: mask normal map by view direction to avoid skip value
                    H, W = data.image_height, data.image_width
                    # view_dirs = -(
                    #     (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
                    #     .sum(dim=-1)
                    #     .reshape(H, W, 3)
                    # )  # [H, W, 3]
                    view_dirs = -get_world_camera_rays_from_intrinsic(data).reshape(H, W, 3)

                    # if self.occlusion_flag and self.GS_config.enable_occ:
                    #     for i in range(9):
                    #         filepath = os.path.join(self.GS_config.occ_dir, f"occlusion_volumes_{i}.pth")
                    #         print(f"begin to load occlusion volumes from {filepath}")
                    #         occlusion_volumes = torch.load(filepath)
                    #         self.occlusion_ids_list.append(occlusion_volumes["occlusion_ids"])
                    #         self.occlusion_coefficients_list.append(occlusion_volumes["occlusion_coefficients"])
                    #         self.occlusion_degree = occlusion_volumes["degree"]
                    #         self.bound = occlusion_volumes["bound"]
                    #         self.aabb = torch.tensor(
                    #             [-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).cuda()
                    #     self.occlusion_flag = False
                    # # recon occlusion
                    # if self.GS_config.enable_occ:
                    #     # TODO if we can get the index of the most contributing point for each pixel, then we can use the index to get to the deformed space
                    #     points = (
                    #         (-view_dirs.reshape(-1, 3) * render_pkg["depth_map"].reshape(-1, 1) + c2w[:3, 3])
                    #         .contiguous()
                    #     )  # [HW, 3]
                    #     # should similar to render_pkg["deformed_gaussian"].get_xyz
                    #     # os.makedirs(os.path.join(self.GS_config.exp_dir, 'test_occ'), exist_ok=True)
                    #     # use_plyfile(points, os.path.join(self.GS_config.exp_dir, 'test_occ', f"points.ply"))
                    #     # use_plyfile(render_pkg["deformed_gaussian"].get_xyz, os.path.join(self.GS_config.exp_dir, 'test_occ', f"dp.ply"))
                    #     # TODO check whether those points is aligned with the point cloud!!!!!!!!!!!
                    #     part_joint = [16, 18, 17, 19, 2, 5, 1, 4, 0]
                    #     occlusion_list = []
                    #     for i in range(9):
                    #         tfs = data.bone_transforms  # [B, 4, 4]
                    #         tfs_part = tfs[part_joint[i]]
                    #         n_pts = points.shape[0]
                    #         homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=points.device)
                    #         x_hat_homo = torch.cat([points, homo_coord], dim=-1).view(n_pts, 4, 1)
                    #         inv_tfs = torch.linalg.inv(tfs_part).reshape(1, 4, 4)
                    #         x_c = torch.matmul(inv_tfs, x_hat_homo)[:, :3, 0]
                    #         n_c = torch.matmul(inv_tfs[:, :3, :3], normal_map.permute(1, 2, 0).reshape(-1, 3, 1))[:, :3,
                    #               0]
                    #         occlusion = recon_occlusion(
                    #             H=H,
                    #             W=W,
                    #             bound=self.bound,
                    #             points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
                    #             normals=n_c.contiguous(),
                    #             occlusion_coefficients=self.occlusion_coefficients_list[i],
                    #             occlusion_ids=self.occlusion_ids_list[i],
                    #             aabb=self.aabb,
                    #             degree=self.occlusion_degree,
                    #         ).reshape(H, W, 1)
                    #         occlusion_list.append(occlusion)
                    #         # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))
                    #
                    #     occlusion = torch.stack(occlusion_list, dim=-1).prod(dim=-1)
                    #     # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
                    #     irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                    # else:
                    #     occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                    #     irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

                    if self.occlusion_flag and (self.GS_config.enable_occ_type == 'pixel' or self.GS_config.enable_occ_type == 'gaussian'):
                        for i in range(9):
                            filepath = os.path.join(self.GS_config.occ_dir, f"occlusion_volumes_{i}.pth")
                            print(f"begin to load occlusion volumes from {filepath}")
                            occlusion_volumes = torch.load(filepath)
                            self.occlusion_ids_list.append(occlusion_volumes["occlusion_ids"])
                            self.occlusion_coefficients_list.append(occlusion_volumes["occlusion_coefficients"])
                            self.occlusion_degree = occlusion_volumes["degree"]
                            self.bound = occlusion_volumes["bound"]
                            self.aabb = torch.tensor(
                                    [-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).cuda()
                        self.occlusion_flag = False
                    # recon occlusion
                    if self.GS_config.enable_occ_type == 'pixel':
                        # TODO if we can get the index of the most contributing point for each pixel, then we can use the index to get to the deformed space
                        points = (
                            (-view_dirs.reshape(-1, 3) * render_pkg["depth_map"].reshape(-1, 1) + c2w[:3, 3])
                            .contiguous()
                        )  # [HW, 3]
                        # should similar to render_pkg["deformed_gaussian"].get_xyz
                        # os.makedirs(os.path.join(self.GS_config.exp_dir, 'test_occ'), exist_ok=True)
                        # use_plyfile(points, os.path.join(self.GS_config.exp_dir, 'test_occ', f"points.ply"))
                        # use_plyfile(render_pkg["deformed_gaussian"].get_xyz, os.path.join(self.GS_config.exp_dir, 'test_occ', f"dp.ply"))
                        # TODO check whether those points is aligned with the point cloud!!!!!!!!!!!
                        part_joint = [16, 18, 17, 19, 2, 5, 1, 4, 0]
                        occlusion_list = []
                        for i in range(9):
                            tfs = data.bone_transforms  # [B, 4, 4]
                            tfs_part = tfs[part_joint[i]]
                            n_pts = points.shape[0]
                            homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=points.device)
                            x_hat_homo = torch.cat([points, homo_coord], dim=-1).view(n_pts, 4, 1)
                            inv_tfs = torch.linalg.inv(tfs_part).reshape(1, 4, 4)
                            x_c = torch.matmul(inv_tfs, x_hat_homo)[:, :3, 0]
                            n_c = torch.matmul(inv_tfs[:, :3, :3], normal_map.permute(1, 2, 0).reshape(-1, 3, 1))[:, :3,
                                  0]
                            occlusion = recon_occlusion(
                                H=H,
                                W=W,
                                bound=self.bound,
                                points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
                                normals=n_c.contiguous(),
                                occlusion_coefficients=self.occlusion_coefficients_list[i],
                                occlusion_ids=self.occlusion_ids_list[i],
                                aabb=self.aabb,
                                degree=self.occlusion_degree,
                            ).reshape(H, W, 1)
                            occlusion_list.append(occlusion)
                            # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

                        occlusion = torch.stack(occlusion_list, dim=-1).prod(dim=-1)
                        # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
                        irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                    elif self.GS_config.enable_occ_type == 'gaussian':
                        # print("begin to use gaussian splatting to render occlusion")
                        # deformed normal
                        normal_points = render_pkg["normal_points"]
                        points = render_pkg["deformed_gaussian"].get_xyz

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
                            occlusion = recon_occlusion(
                                H=points.shape[0],
                                W=1,
                                bound=self.bound,
                                points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
                                normals=n_c.contiguous(),
                                occlusion_coefficients=self.occlusion_coefficients_list[i],
                                occlusion_ids=self.occlusion_ids_list[i],
                                aabb=self.aabb,
                                degree=self.occlusion_degree,
                            )
                            # import pdb;pdb.set_trace()
                            occlusion = occlusion.reshape(points.shape[0])
                            part_occlusion_list.append(occlusion)
                            # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

                        # calculate self occ
                        # TODO we can calculate it outside the loop
                        points = scene.gaussians.get_xyz
                        normal_points = self.scene.gaussians.get_normal
                        # real_rotation = build_rotation(scene.gaussians._rotation)
                        # real_scales = scene.gaussians.get_scaling
                        # dim1_index = torch.arange(0, real_scales.shape[0], device=real_scales.device)
                        # small_index = torch.argmin(real_scales, dim=-1)
                        # rotation_vec = real_rotation[dim1_index, :, small_index]
                        # normal_points = rotation_vec / torch.norm(rotation_vec, dim=-1, keepdim=True)
                        self_occlusion_list = []
                        for i in range(9):
                            occlusion = recon_occlusion(
                                H=points.shape[0],
                                W=1,
                                bound=self.bound,
                                points=points.clamp(min=-self.bound, max=self.bound).contiguous(),
                                normals=normal_points.contiguous(),
                                occlusion_coefficients=self.occlusion_coefficients_list[i],
                                occlusion_ids=self.occlusion_ids_list[i],
                                aabb=self.aabb,
                                degree=self.occlusion_degree,
                            )
                            # import pdb;pdb.set_trace()
                            occlusion = occlusion.reshape(points.shape[0])
                            self_occlusion_list.append(occlusion)

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
                            point_mask = points_weights_part_idx.reshape(points_weights_part_idx.shape[0],
                                                                         1) == part.reshape(1, -1)
                            point_mask = point_mask.any(dim=1)
                            points_group_one_hot[point_mask, idx] = 1

                        part_occlusion_list = torch.stack(part_occlusion_list, dim=-1)  # [N, 9]
                        self_occlusion_list = torch.stack(self_occlusion_list, dim=-1)  # [N, 9]
                        occlusion = torch.where(
                            points_group_one_hot > 0.5,
                            self_occlusion_list,
                            part_occlusion_list
                        )
                        # import pdb; pdb.set_trace()
                        # combine_all_type_occlusion
                        occlusion_per_gs = occlusion.prod(dim=-1)  # [N]
                        # gasussian splatting
                        # gasussian splatting
                        if self.rendering_type == 'forward_pbr':
                            occlusion = occlusion_per_gs
                            irradiance = torch.zeros_like(occlusion)  # [N, 1]
                        else:
                            occlusion = render_fast(occlusion_per_gs, render_pkg["deformed_gaussian"], data,
                                                    self.GS_config.opt.iterations, self.scene,
                                                    self.GS_config.pipeline, self.background)['render'].permute(1, 2,
                                                                                                                0).clamp(
                                min=0.0, max=1.0)
                            irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                        # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
                        # occlusion_img = occlusion.clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        # torchvision.utils.save_image(occlusion_img, os.path.join(occlusion_path, f"occlusion_{view.image_name}.png"))
                        # wandb_img = wandb.Image(occlusion_img[None], caption=config['name'] + "_view_{}/occlusion".format(data.image_name))
                        # examples.append(wandb_img)

                    else:
                        occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                        irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

                    normal_mask = render_pkg["normal_mask"]  # [1, H, W]
                    self.scene.cubemap.build_mips()  # build mip for environment light

                    if self.rendering_type == 'forward_pbr':
                        pbr_result_gs = pbr_shading_gs(deformed_gaussian=render_pkg["deformed_gaussian"],
                                                       camera_center=c2w[:3, 3],
                                                       # TODO check if the camera center is correct c2w[:3, 3]
                                                       light=self.scene.cubemap,
                                                       occlusion=occlusion,  # [pc, 1]
                                                       irradiance=irradiance,  # [pc, 1]
                                                       brdf_lut=self.brdf_lut)
                        pbr_result = \
                        render_fast(pbr_result_gs, render_pkg["deformed_gaussian"], data, self.GS_config.opt.iterations,
                                    self.scene,
                                    self.GS_config.pipeline, self.background, inference=False)['render'].clamp(min=0.0,
                                                                                                               max=1.0)  # [3, H, W]
                        render_rgb = linear_to_srgb(pbr_result)
                        # TODO check whether it is needed
                        render_rgb = torch.where(
                            normal_mask,
                            render_rgb,
                            self.background[:, None, None],
                        )
                        # TODO update weights
                        # pbr_render_loss = l1_loss(render_rgb, gt_image)
                        # loss += pbr_render_loss
                        wandb_img = wandb.Image(render_rgb[None],
                                                caption=config['name'] + "_view_{}/pbr_image".format(data.image_name))
                        examples.append(wandb_img)
                    else:
                        pbr_result = pbr_shading(
                            light=self.scene.cubemap,
                            normals=normal_map.permute(1, 2, 0).detach(),  # [H, W, 3]
                            view_dirs=view_dirs,
                            mask=render_pkg["opacity_mask"].permute(1, 2, 0),  # [H, W, 1]
                            albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                            metallic=metallic_map.permute(1, 2, 0),  # [H, W, 1]
                            tone=False,
                            gamma=True,
                            occlusion=occlusion,
                            irradiance=irradiance,
                            brdf_lut=self.brdf_lut,
                        )
                        # render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
                        # # TODO check whether it is needed
                        # render_rgb = torch.where(
                        #     normal_mask,
                        #     render_rgb,
                        #     self.background[:, None, None],
                        # )

                        diffuse_rgb = (
                            pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        specular_rgb = (
                            pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        render_rgb = (
                            pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                        )  # [3, H, W]
                        # NOTE: mask render_rgb by depth map
                        background = renderArgs[1]
                        render_rgb = torch.where(
                            # render_pkg["opacity_mask"],
                            normal_mask,
                            render_rgb,
                            background[:, None, None],
                        )
                        diffuse_rgb = torch.where(
                            # render_pkg["opacity_mask"],
                            normal_mask,
                            diffuse_rgb,
                            background[:, None, None],
                        )
                        specular_rgb = torch.where(
                            # render_pkg["opacity_mask"],
                            normal_mask,
                            specular_rgb,
                            background[:, None, None],
                        )
                        pbr_image = torch.cat(
                            [render_rgb, diffuse_rgb, specular_rgb], dim=2
                        )  # [3, H, 3W]

                        wandb_img = wandb.Image(pbr_image[None],
                                                caption=config['name'] + "_view_{}/pbr_image".format(data.image_name))
                        examples.append(wandb_img)

                    wandb.log({config['name'] + "_images": examples})
                    examples.clear()

                    l1_test_pbr += l1_loss(render_rgb, gt_image).mean().double()
                    metrics_test_pbr = evaluator(
                        render_rgb,
                        gt_image,
                        valid_mask=data.valid_msk,
                    )
                    psnr_test_pbr += metrics_test_pbr["psnr"]
                    ssim_test_pbr += metrics_test_pbr["ssim"]
                    lpips_test_pbr += metrics_test_pbr["lpips"]


                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                psnr_test_pbr /= len(config['cameras'])
                ssim_test_pbr /= len(config['cameras'])
                lpips_test_pbr /= len(config['cameras'])
                l1_test_pbr /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("\n[ITER {}] Evaluating {}: L1_PBR {} PSNR_PBR {}".format(iteration, config['name'], l1_test_pbr, psnr_test_pbr))
                wandb.log({
                    config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                    config['name'] + '/loss_viewpoint - psnr': psnr_test,
                    config['name'] + '/loss_viewpoint - ssim': ssim_test,
                    config['name'] + '/loss_viewpoint - lpips': lpips_test,
                    config['name'] + '/loss_viewpoint_pbr - l1_loss': l1_test_pbr,
                    config['name'] + '/loss_viewpoint_pbr - psnr': psnr_test_pbr,
                    config['name'] + '/loss_viewpoint_pbr - ssim': ssim_test_pbr,
                    config['name'] + '/loss_viewpoint_pbr - lpips': lpips_test_pbr,
                })

        wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
        wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
        torch.cuda.empty_cache()
        scene.train()


    def forward_GS_pbr_step(self, data_idx):

        data = self.scene.train_dataset[data_idx]
        self.iter += 1
        self.iter_start.record()
        self.gaussians.update_learning_rate(self.iter)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iter % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # NOTE: black background for PBR? we always use black background

        lambda_mask = self.GS_C(self.iter, self.GS_config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, self.iter, self.scene, self.pipe, self.background, compute_loss=True,
                            return_opacity=use_mask, return_normal_points=True, derive_normal=True)

        return render_pkg, data

    def forward_GS_step(self, data_idx):
        # if not self.data_stack:
        #     self.data_stack = list(range(len(self.scene.train_dataset)))
        # data_idx = self.data_stack.pop(randint(0, len(self.data_stack) - 1))
        data = self.scene.train_dataset[data_idx]
        self.iter += 1
        self.iter_start.record()
        self.gaussians.update_learning_rate(self.iter)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if self.iter % 1000 == 0:
            self.gaussians.oneupSHdegree()

        lambda_mask = self.GS_C(self.iter, self.GS_config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, self.iter, self.scene, self.pipe, self.background, compute_loss=True,
                            return_opacity=use_mask, return_normal_points=True)

        return render_pkg, data

    def forward_GS_fast(self, data_idx):
        # if not self.data_stack:
        #     self.data_stack = list(range(len(self.scene.train_dataset)))
        # data_idx = self.data_stack.pop(randint(0, len(self.data_stack) - 1))
        data = self.scene.train_dataset[data_idx]
        pc, loss_reg, colors_precomp = self.scene.convert_gaussians(data, self.iter, compute_loss=False)
        return pc, data


    def optimize_GS_pbr_step(self, render_pkg, data, distill_data):

        # c2w = torch.inverse(data.world_view_transform.T)  # [4, 4]
        # w2c = data.world_view_transform.T # [4, 4]
        c2w = torch.inverse(data.w2c_opencv)
        w2c = data.w2c_opencv
        # canonical_rays = get_canonical_rays_from_intrinsic(data)

        image = render_pkg["render"]  # [3, H, W]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth_map = render_pkg["depth_map"]  # [1, H, W]
        normal_map_from_depth = render_pkg["normal_map_from_depth"]  # [3, H, W]
        normal_map = render_pkg["normal_map"]  # [3, H, W]
        albedo_map = render_pkg["albedo_map"]  # [3, H, W]
        roughness_map = render_pkg["roughness_map"]  # [1, H, W]
        metallic_map = render_pkg["metallic_map"]  # [1, H, W]

        # # formulate roughness
        # rmax, rmin = 1.0, 0.04
        # roughness_map = roughness_map * (rmax - rmin) + rmin

        # NOTE: mask normal map by view direction to avoid skip value
        H, W = data.image_height, data.image_width
        # view_dirs = -(
        #     (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
        #     .sum(dim=-1)
        #     .reshape(H, W, 3)
        # )  # [H, W, 3]
        view_dirs = -get_world_camera_rays_from_intrinsic(data).reshape(H, W, 3)

        lambda_mask = self.GS_C(self.iter, self.GS_config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        opacity = render_pkg["opacity_render"] if use_mask else None

        # Loss
        gt_image = data.original_image.cuda()

        lambda_l1 = self.GS_C(self.iter, self.GS_config.opt.lambda_l1)
        lambda_dssim = self.GS_C(self.iter, self.GS_config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = self.GS_C(self.iter, self.GS_config.opt.get('lambda_perceptual', 0.))
        if lambda_perceptual > 0:
            # crop the foreground
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]

            loss_perceptual = self.loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif self.GS_config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif self.GS_config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = self.GS_C(self.iter, self.GS_config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = self.scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = self.GS_C(self.iter, self.GS_config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = self.GS_C(self.iter, self.GS_config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(self.scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # scaling regularization
        scaling = render_pkg["deformed_gaussian"].get_scaling
        min_scaling = torch.min(torch.abs(scaling), dim=-1)[0].mean()
        lambda_scale = self.GS_C(self.iter, self.GS_config.opt.get('lambda_scale', 0.))
        loss += lambda_scale * min_scaling

        # distillation normal
        loss_normal_similarity = torch.tensor(0.).cuda()
        lambda_normal_similarity = self.GS_C(self.iter, self.GS_config.opt.get('lambda_normal_similarity', 0.))
        if distill_data is not None and lambda_normal_similarity > 0.:
            valid = distill_data['valid'].detach()
            normal_world_IA = distill_data['normal_world'].detach()
            normal_points_GS = render_pkg['normal_points']
            loss_normal_similarity = -F.cosine_similarity(normal_world_IA[valid], normal_points_GS[valid],
                                                          dim=-1).mean()
            # TODO check the shape of the similarity loss
            loss += lambda_normal_similarity * loss_normal_similarity

        # distillation position
        loss_pos_distill = torch.tensor(0.).cuda()
        lambda_pos_distill = self.GS_C(self.iter, self.GS_config.opt.get('lambda_pos_distill', 0.))
        if distill_data is not None and lambda_pos_distill > 0.:
            sdf = distill_data['sdf'].reshape(-1, 1).detach()
            valid = distill_data['valid'].detach()
            normal_world_IA = distill_data['normal_world'].detach()
            target_positions = (normal_world_IA * -sdf) + render_pkg['deformed_gaussian'].get_xyz
            # compute MSE loss
            loss_pos_distill = F.mse_loss(render_pkg['deformed_gaussian'].get_xyz[valid], target_positions[valid])
            # TODO check the shape of the mse loss
            loss += lambda_pos_distill * loss_pos_distill

        # TODO distillation albedo
        loss_materials = torch.tensor(0.).cuda()
        lambda_materials = self.GS_C(self.iter, self.GS_config.opt.get('lambda_materials', 0.))
        if distill_data is not None and lambda_materials > 0.:
            valid = distill_data['valid'].detach()
            materials_IA = distill_data['materials'].detach()
            albedo_GS = render_pkg['deformed_gaussian'].get_albedo
            roughness_GS = render_pkg['deformed_gaussian'].get_roughness
            metallic_GS = render_pkg['deformed_gaussian'].get_metallic
            roughness_GS = self.scene.distill_adapt_roughness_layer(roughness_GS.reshape(1, -1)).reshape(-1, 1)
            metallic_GS = self.scene.distill_adapt_metallic_layer(metallic_GS.reshape(1, -1)).reshape(-1, 1)
            loss_materials += F.mse_loss(albedo_GS[valid], materials_IA[valid, :3])
            loss_materials += F.mse_loss(roughness_GS[valid], materials_IA[valid, 3:4])
            loss_materials += F.mse_loss(metallic_GS[valid], materials_IA[valid, 4:])
            # TODO check the shape and order of the materials_IA
            loss += lambda_materials * loss_materials

        # distill env_map
        envmap = dr.texture(
            self.scene.cubemap.base[None, ...],
            self.envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        loss_envmap = torch.tensor(0.).cuda()
        lambda_env = self.GS_C(self.iter, self.GS_config.opt.get('lambda_env', 0.))
        if distill_data is not None and lambda_env > 0.:
            loss_envmap = F.mse_loss(distill_data['env_map'].detach(), envmap.reshape(-1, 3))
            loss += loss_envmap * lambda_env

        # distill image-based normal & depth
        loss_image_based = torch.tensor(0.).cuda()
        lambda_distill_image = self.GS_C(self.iter, self.GS_config.opt.get('lambda_distill_image', 0.))
        if data.normal_img is not None and lambda_distill_image > 0.:
            # distill data.normal_img
            # convert render_pkg["normal_map"] to normal_img
            gl_normal_img = self.transform_normals(w2c, render_pkg["normal_map"]) * 0.5 + 0.5
            # import pdb; pdb.set_trace()
            # TODO check if gl_normal_img looks like data.normal_img
            # import torchvision
            # torchvision.utils.save_image(gl_normal_img, './gl_normal_img.png')
            # torchvision.utils.save_image(data.normal_img, './data_normal_img.png')
            valid_image_mask = data.original_mask.reshape(-1) > 0.5
            # torchvision.utils.save_image(data.original_mask, './data_original_mask.png')
            # torchvision.utils.save_image(data.original_image, './data_original_image.png')
            loss_image_based = l1_loss(gl_normal_img.reshape(3, -1)[:, valid_image_mask], data.normal_img.reshape(3, -1)[:, valid_image_mask])
            # distill data.depth_value
            # todo here the render_pkg["depth_map"] is t, we need to check whether data.depth_value is also t
            loss_image_based += l1_loss(render_pkg["depth_map"].reshape(-1)[valid_image_mask], data.depth_value.reshape(-1)[valid_image_mask])
            loss += loss_image_based * lambda_distill_image



        pbr_render_loss = torch.tensor(0.).cuda()
        brdf_tv_loss = torch.tensor(0.).cuda()
        lamb_loss = torch.tensor(0.).cuda()
        env_tv_loss = torch.tensor(0.).cuda()
        if self.iter > self.pbr_iteration:

            # if self.occlusion_flag and self.GS_config.enable_occ:
            #     for i in range(9):
            #         filepath = os.path.join(self.GS_config.occ_dir, f"occlusion_volumes_{i}.pth")
            #         print(f"begin to load occlusion volumes from {filepath}")
            #         occlusion_volumes = torch.load(filepath)
            #         self.occlusion_ids_list.append(occlusion_volumes["occlusion_ids"])
            #         self.occlusion_coefficients_list.append(occlusion_volumes["occlusion_coefficients"])
            #         self.occlusion_degree = occlusion_volumes["degree"]
            #         self.bound = occlusion_volumes["bound"]
            #         self.aabb = torch.tensor([-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).cuda()
            #     self.occlusion_flag = False
            # # recon occlusion
            # if self.GS_config.enable_occ:
            #     # TODO if we can get the index of the most contributing point for each pixel, then we can use the index to get to the deformed space
            #     points = (
            #         (-view_dirs.reshape(-1, 3) * depth_map.reshape(-1, 1) + c2w[:3, 3])
            #         .contiguous()
            #     )  # [HW, 3]
            #     # should similar to render_pkg["deformed_gaussian"].get_xyz
            #     # os.makedirs(os.path.join(self.GS_config.exp_dir, 'test_occ'), exist_ok=True)
            #     # use_plyfile(points, os.path.join(self.GS_config.exp_dir, 'test_occ', f"points.ply"))
            #     # use_plyfile(render_pkg["deformed_gaussian"].get_xyz, os.path.join(self.GS_config.exp_dir, 'test_occ', f"dp.ply"))
            #     # TODO check whether those points is aligned with the point cloud!!!!!!!!!!!
            #     part_joint = [16, 18, 17, 19, 2, 5, 1, 4, 0]
            #     # part_group = [
            #     #     [16],
            #     #     [18, 20, 22],
            #     #     [17],
            #     #     [19, 21, 23],
            #     #     [2, ],
            #     #     [5, 8, 11],
            #     #     [1, ],
            #     #     [4, 7, 10],
            #     #     [0, 3, 6, 9, 13, 14, 12, 15],
            #     # ]
            #     occlusion_list = []
            #     for i in range(9):
            #         tfs = data.bone_transforms # [B, 4, 4]
            #         tfs_part = tfs[part_joint[i]]
            #         n_pts = points.shape[0]
            #         homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=points.device)
            #         x_hat_homo = torch.cat([points, homo_coord], dim=-1).view(n_pts, 4, 1)
            #         inv_tfs = torch.linalg.inv(tfs_part).reshape(1, 4, 4)
            #         x_c = torch.matmul(inv_tfs, x_hat_homo)[:, :3, 0]
            #         n_c = torch.matmul(inv_tfs[:, :3, :3], normal_map.permute(1, 2, 0).reshape(-1, 3, 1))[:, :3, 0]
            #         occlusion = recon_occlusion(
            #             H=H,
            #             W=W,
            #             bound=self.bound,
            #             points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
            #             normals=n_c.contiguous(),
            #             occlusion_coefficients=self.occlusion_coefficients_list[i],
            #             occlusion_ids=self.occlusion_ids_list[i],
            #             aabb=self.aabb,
            #             degree=self.occlusion_degree,
            #         ).reshape(H, W, 1)
            #         occlusion_list.append(occlusion)
            #
            #         # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))
            #
            #     occlusion = torch.stack(occlusion_list, dim=-1).prod(dim=-1)
            #     # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
            #     # import pdb; pdb.set_trace()
            #     irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
            #     # irradiance = irradiance_volumes.query_irradiance(
            #     #     points=points.reshape(-1, 3).contiguous(),
            #     #     normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
            #     # ).reshape(H, W, -1)
            # else:
            #     occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
            #     irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
            if self.occlusion_flag and (
                    self.GS_config.enable_occ_type == 'pixel' or self.GS_config.enable_occ_type == 'gaussian'):
                for i in range(9):
                    filepath = os.path.join(self.GS_config.occ_dir, f"occlusion_volumes_{i}.pth")
                    print(f"begin to load occlusion volumes from {filepath}")
                    occlusion_volumes = torch.load(filepath)
                    self.occlusion_ids_list.append(occlusion_volumes["occlusion_ids"])
                    self.occlusion_coefficients_list.append(occlusion_volumes["occlusion_coefficients"])
                    self.occlusion_degree = occlusion_volumes["degree"]
                    self.bound = occlusion_volumes["bound"]
                    self.aabb = torch.tensor(
                        [-self.bound, -self.bound, -self.bound, self.bound, self.bound, self.bound]).cuda()
                self.occlusion_flag = False
            # recon occlusion
            if self.GS_config.enable_occ_type == 'pixel':
                # TODO if we can get the index of the most contributing point for each pixel, then we can use the index to get to the deformed space
                points = (
                    (-view_dirs.reshape(-1, 3) * render_pkg["depth_map"].reshape(-1, 1) + c2w[:3, 3])
                    .contiguous()
                )  # [HW, 3]
                # should similar to render_pkg["deformed_gaussian"].get_xyz
                # os.makedirs(os.path.join(self.GS_config.exp_dir, 'test_occ'), exist_ok=True)
                # use_plyfile(points, os.path.join(self.GS_config.exp_dir, 'test_occ', f"points.ply"))
                # use_plyfile(render_pkg["deformed_gaussian"].get_xyz, os.path.join(self.GS_config.exp_dir, 'test_occ', f"dp.ply"))
                # TODO check whether those points is aligned with the point cloud!!!!!!!!!!!
                part_joint = [16, 18, 17, 19, 2, 5, 1, 4, 0]
                occlusion_list = []
                for i in range(9):
                    tfs = data.bone_transforms  # [B, 4, 4]
                    tfs_part = tfs[part_joint[i]]
                    n_pts = points.shape[0]
                    homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=points.device)
                    x_hat_homo = torch.cat([points, homo_coord], dim=-1).view(n_pts, 4, 1)
                    inv_tfs = torch.linalg.inv(tfs_part).reshape(1, 4, 4)
                    x_c = torch.matmul(inv_tfs, x_hat_homo)[:, :3, 0]
                    n_c = torch.matmul(inv_tfs[:, :3, :3], normal_map.permute(1, 2, 0).reshape(-1, 3, 1))[:, :3,
                          0]
                    occlusion = recon_occlusion(
                        H=H,
                        W=W,
                        bound=self.bound,
                        points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
                        normals=n_c.contiguous(),
                        occlusion_coefficients=self.occlusion_coefficients_list[i],
                        occlusion_ids=self.occlusion_ids_list[i],
                        aabb=self.aabb,
                        degree=self.occlusion_degree,
                    ).reshape(H, W, 1)
                    occlusion_list.append(occlusion)
                    # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

                occlusion = torch.stack(occlusion_list, dim=-1).prod(dim=-1)
                # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
                irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
            elif self.GS_config.enable_occ_type == 'gaussian':
                # print("begin to use gaussian splatting to render occlusion")
                # deformed normal
                normal_points = render_pkg["normal_points"]
                points = render_pkg["deformed_gaussian"].get_xyz

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
                    occlusion = recon_occlusion(
                        H=points.shape[0],
                        W=1,
                        bound=self.bound,
                        points=x_c.clamp(min=-self.bound, max=self.bound).contiguous(),
                        normals=n_c.contiguous(),
                        occlusion_coefficients=self.occlusion_coefficients_list[i],
                        occlusion_ids=self.occlusion_ids_list[i],
                        aabb=self.aabb,
                        degree=self.occlusion_degree,
                    )
                    # import pdb;pdb.set_trace()
                    occlusion = occlusion.reshape(points.shape[0])
                    part_occlusion_list.append(occlusion)
                    # use_plyfile(x_c, os.path.join(self.GS_config.exp_dir, 'test_occ', f"xc_{i}.ply"))

                # calculate self occ
                # TODO we can calculate it outside the loop
                points = self.scene.gaussians.get_xyz
                normal_points = self.scene.gaussians.get_normal
                # real_rotation = build_rotation(self.scene.gaussians._rotation)
                # real_scales = self.scene.gaussians.get_scaling
                # dim1_index = torch.arange(0, real_scales.shape[0], device=real_scales.device)
                # small_index = torch.argmin(real_scales, dim=-1)
                # rotation_vec = real_rotation[dim1_index, :, small_index]
                # normal_points = rotation_vec / torch.norm(rotation_vec, dim=-1, keepdim=True)
                self_occlusion_list = []
                for i in range(9):
                    occlusion = recon_occlusion(
                        H=points.shape[0],
                        W=1,
                        bound=self.bound,
                        points=points.clamp(min=-self.bound, max=self.bound).contiguous(),
                        normals=normal_points.contiguous(),
                        occlusion_coefficients=self.occlusion_coefficients_list[i],
                        occlusion_ids=self.occlusion_ids_list[i],
                        aabb=self.aabb,
                        degree=self.occlusion_degree,
                    )
                    # import pdb;pdb.set_trace()
                    occlusion = occlusion.reshape(points.shape[0])
                    self_occlusion_list.append(occlusion)

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
                    _, points_weights_smpl = self.scene.converter.deformer.rigid.query_weights(points)
                    points_weights_part_idx = points_weights_smpl.argmax(dim=1)

                    # select point based on the part group
                    part = torch.tensor(part).cuda()
                    point_mask = points_weights_part_idx.reshape(points_weights_part_idx.shape[0],
                                                                 1) == part.reshape(1, -1)
                    point_mask = point_mask.any(dim=1)
                    points_group_one_hot[point_mask, idx] = 1

                part_occlusion_list = torch.stack(part_occlusion_list, dim=-1)  # [N, 9]
                self_occlusion_list = torch.stack(self_occlusion_list, dim=-1)  # [N, 9]
                occlusion = torch.where(
                    points_group_one_hot > 0.5,
                    self_occlusion_list,
                    part_occlusion_list
                )
                # import pdb; pdb.set_trace()
                # combine_all_type_occlusion
                occlusion_per_gs = occlusion.prod(dim=-1)  # [N]
                # gasussian splatting
                if self.rendering_type == 'forward_pbr':
                    occlusion = occlusion_per_gs
                    irradiance = torch.zeros_like(occlusion)  # [N, 1]
                else:
                    occlusion = render_fast(occlusion_per_gs, render_pkg["deformed_gaussian"], data, self.GS_config.opt.iterations, self.scene,
                                            self.GS_config.pipeline, self.background)['render'].permute(1, 2, 0).clamp(min=0.0, max=1.0)
                    irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

                # occlusion = torch.stack(occlusion_list, dim=-1).min(dim=-1)[0]
                # occlusion_img = occlusion.clamp(min=0.0, max=1.0).permute(2, 0, 1)
                # torchvision.utils.save_image(occlusion_img, os.path.join(occlusion_path, f"occlusion_{view.image_name}.png"))
                # wandb_img = wandb.Image(occlusion_img[None], caption=config['name'] + "_view_{}/occlusion".format(data.image_name))
                # examples.append(wandb_img)

            else:
                occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]
                irradiance = torch.zeros_like(roughness_map).permute(1, 2, 0)  # [H, W, 1]

            normal_mask = render_pkg["normal_mask"]  # [1, H, W]
            self.scene.cubemap.build_mips()  # build mip for environment light

            if self.rendering_type == 'forward_pbr':
                pbr_result_gs = pbr_shading_gs(deformed_gaussian=render_pkg["deformed_gaussian"],
                                            camera_center=c2w[:3, 3], # TODO check if the camera center is correct c2w[:3, 3]
                                            light=self.scene.cubemap,
                                            occlusion=occlusion,  # [pc, 1]
                                            irradiance=irradiance,  # [pc, 1]
                                            brdf_lut=self.brdf_lut)
                pbr_result = render_fast(pbr_result_gs, render_pkg["deformed_gaussian"], data, self.GS_config.opt.iterations,
                            self.scene,
                            self.GS_config.pipeline, self.background, inference=False)['render'].clamp(min=0.0, max=1.0) # [3, H, W]
                render_rgb = linear_to_srgb(pbr_result)
                # TODO check whether it is needed
                render_rgb = torch.where(
                    normal_mask,
                    render_rgb,
                    self.background[:, None, None],
                )
                # TODO update weights
                pbr_render_loss = l1_loss(render_rgb, gt_image)
                loss += pbr_render_loss
            else:
                pbr_result = pbr_shading(
                    light=self.scene.cubemap,
                    normals=normal_map.permute(1, 2, 0),
                    # normals=normal_map.permute(1, 2, 0).detach(),  # [H, W, 3]
                    view_dirs=view_dirs,
                    mask=render_pkg["opacity_mask"].permute(1, 2, 0),  # [H, W, 1]
                    albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                    roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                    metallic=metallic_map.permute(1, 2, 0),  # [H, W, 1]
                    tone=False,
                    gamma=True,
                    occlusion=occlusion,
                    irradiance=irradiance,
                    brdf_lut=self.brdf_lut,
                )
                render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
                # TODO check whether it is needed
                render_rgb = torch.where(
                    normal_mask,
                    render_rgb,
                    self.background[:, None, None],
                )
                # TODO update weights
                pbr_render_loss = l1_loss(render_rgb, gt_image)
                loss += pbr_render_loss

            ### BRDF loss
            if (normal_mask == 0).sum() > 0:
                brdf_tv_loss = self.get_masked_tv_loss(
                    normal_mask,
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                )
            else:
                brdf_tv_loss = self.get_tv_loss(
                    gt_image,  # [3, H, W]
                    torch.cat([albedo_map, roughness_map, metallic_map], dim=0),  # [5, H, W]
                    pad=1,  # FIXME: 8 for scene
                    step=1,
                )
            brdf_tv_weight = 1.0
            loss += brdf_tv_loss * brdf_tv_weight
            lamb_weight = 0.001
            lamb_loss = (1.0 - roughness_map[normal_mask]).mean() + metallic_map[normal_mask].mean()
            loss += lamb_loss * lamb_weight

            #### envmap
            # TV smoothness
            tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            env_tv_loss = tv_h1 + tv_w1
            env_tv_weight = 0.01
            loss += env_tv_loss * env_tv_weight


        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = self.opt.get(f"lambda_{name}", 0.)
            lbd = self.GS_C(self.iter, lbd)
            loss += lbd * value
        loss.backward()

        self.iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = self.iter_start.elapsed_time(self.iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/min_scaling': min_scaling.item(),
                'loss/normal_similarity': loss_normal_similarity.item(),
                'loss/pos_distill': loss_pos_distill.item(),
                'loss/materials': loss_materials.item(),
                'loss/envmap': loss_envmap.item(),
                'loss/image_based': loss_image_based.item(),
                'loss/pbr_render_loss': pbr_render_loss.item(),
                'loss/brdf_tv_loss': brdf_tv_loss.item(),
                'loss/lamb_loss': lamb_loss.item(),
                'loss/env_tv_loss': env_tv_loss.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iter % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iter == self.opt.iterations:
                self.progress_bar.close()

            # Log and save
            self.validation_pbr(self.iter, self.testing_iterations, self.testing_interval, self.scene, self.evaluator,
                            (self.pipe, self.background))
            if (self.iter in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iter))
                self.scene.save(self.iter)

            # Densification
            if self.iter < self.opt.densify_until_iter and self.iter > self.model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iter > self.opt.densify_from_iter and self.iter % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iter > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt, self.scene, size_threshold)

                if self.iter % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iter == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iter < self.opt.iterations:
                self.scene.optimize(self.iter)

            if self.iter in self.checkpoint_iterations:
                self.scene.save_checkpoint(self.iter)
    def optimize_GS_step(self, render_pkg, data, distill_data):
        lambda_mask = self.GS_C(self.iter, self.GS_config.opt.lambda_mask)
        use_mask = lambda_mask > 0.

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None

        # Loss
        gt_image = data.original_image.cuda()

        lambda_l1 = self.GS_C(self.iter, self.GS_config.opt.lambda_l1)
        lambda_dssim = self.GS_C(self.iter, self.GS_config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = self.GS_C(self.iter, self.GS_config.opt.get('lambda_perceptual', 0.))
        if lambda_perceptual > 0:
            # crop the foreground
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]

            loss_perceptual = self.loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif self.GS_config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif self.GS_config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # skinning loss
        lambda_skinning = self.GS_C(self.iter, self.GS_config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = self.scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = self.GS_C(self.iter, self.GS_config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = self.GS_C(self.iter, self.GS_config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(self.scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # scaling regularization
        scaling = render_pkg["deformed_gaussian"].get_scaling
        min_scaling = torch.min(torch.abs(scaling), dim=-1)[0].mean()
        lambda_scale = self.GS_C(self.iter, self.GS_config.opt.get('lambda_scale', 0.))
        loss += lambda_scale * min_scaling

        # distillation
        loss_normal_similarity = torch.tensor(0.).cuda()
        lambda_normal_similarity = self.GS_C(self.iter, self.GS_config.opt.get('lambda_normal_similarity', 0.))
        if distill_data is not None and lambda_normal_similarity > 0.:
            valid = distill_data['valid'].detach()
            normal_world_IA = distill_data['normal_world'].detach()
            normal_points_GS = render_pkg['normal_points']
            loss_normal_similarity = -F.cosine_similarity(normal_world_IA[valid], normal_points_GS[valid], dim=-1).mean()
            # TODO check the shape of the similarity loss
            loss += lambda_normal_similarity * loss_normal_similarity

        loss_pos_distill = torch.tensor(0.).cuda()
        lambda_pos_distill = self.GS_C(self.iter, self.GS_config.opt.get('lambda_pos_distill', 0.))
        if distill_data is not None and lambda_pos_distill > 0.:
            sdf = distill_data['sdf'].reshape(-1, 1).detach()
            valid = distill_data['valid'].detach()
            normal_world_IA = distill_data['normal_world'].detach()
            target_positions = (normal_world_IA * -sdf) + render_pkg['deformed_gaussian'].get_xyz
            # compute MSE loss
            loss_pos_distill = F.mse_loss(render_pkg['deformed_gaussian'].get_xyz[valid], target_positions[valid])
            # TODO check the shape of the mse loss
            loss += lambda_pos_distill * loss_pos_distill

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = self.opt.get(f"lambda_{name}", 0.)
            lbd = self.GS_C(self.iter, lbd)
            loss += lbd * value
        loss.backward()

        self.iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = self.iter_start.elapsed_time(self.iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/min_scaling': min_scaling.item(),
                'loss/normal_similarity': loss_normal_similarity.item(),
                'loss/pos_distill': loss_pos_distill.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iter % 10 == 0:
                self.progress_bar.set_postfix({"Loss": f"{self.ema_loss_for_log:.{7}f}"})
                self.progress_bar.update(10)
            if self.iter == self.opt.iterations:
                self.progress_bar.close()

            # Log and save
            self.validation(self.iter, self.testing_iterations, self.testing_interval, self.scene, self.evaluator,
                            (self.pipe, self.background))
            if (self.iter in self.saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(self.iter))
                self.scene.save(self.iter)

            # Densification
            if self.iter < self.opt.densify_until_iter and self.iter > self.model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.iter > self.opt.densify_from_iter and self.iter % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iter > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt, self.scene, size_threshold)

                if self.iter % self.opt.opacity_reset_interval == 0 or (
                        self.dataset.white_background and self.iter == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if self.iter < self.opt.iterations:
                self.scene.optimize(self.iter)

            if self.iter in self.checkpoint_iterations:
                self.scene.save_checkpoint(self.iter)


    def train_GS(self):
        # optimize 3D gaussians
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        data_stack = None
        ema_loss_for_log = 0.0
        first_iter = 0
        progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        first_iter += 1
        for iteration in range(first_iter, self.opt.iterations + 1):

            iter_start.record()

            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random data point
            if not data_stack:
                data_stack = list(range(len(self.scene.train_dataset)))
            data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
            data = self.scene.train_dataset[data_idx]

            # Render
            if (iteration - 1) == self.debug_from:
                self.pipe.debug = True

            lambda_mask = self.GS_C(iteration, self.GS_config.opt.lambda_mask)
            use_mask = lambda_mask > 0.
            render_pkg = render(data, iteration, self.scene, self.pipe, self.background, compute_loss=True,
                                return_opacity=use_mask)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            opacity = render_pkg["opacity_render"] if use_mask else None

            # Loss
            gt_image = data.original_image.cuda()

            lambda_l1 = self.GS_C(iteration, self.GS_config.opt.lambda_l1)
            lambda_dssim = self.GS_C(iteration, self.GS_config.opt.lambda_dssim)
            loss_l1 = torch.tensor(0.).cuda()
            loss_dssim = torch.tensor(0.).cuda()
            if lambda_l1 > 0.:
                loss_l1 = l1_loss(image, gt_image)
            if lambda_dssim > 0.:
                loss_dssim = 1.0 - ssim(image, gt_image)
            loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

            # perceptual loss
            lambda_perceptual = self.GS_C(iteration, self.GS_config.opt.get('lambda_perceptual', 0.))
            if lambda_perceptual > 0:
                # crop the foreground
                mask = data.original_mask.cpu().numpy()
                mask = np.where(mask)
                y1, y2 = mask[1].min(), mask[1].max() + 1
                x1, x2 = mask[2].min(), mask[2].max() + 1
                fg_image = image[:, y1:y2, x1:x2]
                gt_fg_image = gt_image[:, y1:y2, x1:x2]

                loss_perceptual = self.loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
                loss += lambda_perceptual * loss_perceptual
            else:
                loss_perceptual = torch.tensor(0.)

            # mask loss
            gt_mask = data.original_mask.cuda()
            if not use_mask:
                loss_mask = torch.tensor(0.).cuda()
            elif self.GS_config.opt.mask_loss_type == 'bce':
                opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
                loss_mask = F.binary_cross_entropy(opacity, gt_mask)
            elif self.GS_config.opt.mask_loss_type == 'l1':
                loss_mask = F.l1_loss(opacity, gt_mask)
            else:
                raise ValueError
            loss += lambda_mask * loss_mask

            # skinning loss
            lambda_skinning = self.GS_C(iteration, self.GS_config.opt.lambda_skinning)
            if lambda_skinning > 0:
                loss_skinning = self.scene.get_skinning_loss()
                loss += lambda_skinning * loss_skinning
            else:
                loss_skinning = torch.tensor(0.).cuda()

            lambda_aiap_xyz = self.GS_C(iteration, self.GS_config.opt.get('lambda_aiap_xyz', 0.))
            lambda_aiap_cov = self.GS_C(iteration, self.GS_config.opt.get('lambda_aiap_cov', 0.))
            if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
                loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(self.scene.gaussians, render_pkg["deformed_gaussian"])
            else:
                loss_aiap_xyz = torch.tensor(0.).cuda()
                loss_aiap_cov = torch.tensor(0.).cuda()
            loss += lambda_aiap_xyz * loss_aiap_xyz
            loss += lambda_aiap_cov * loss_aiap_cov

            # scaling regularization
            scaling = render_pkg["deformed_gaussian"].get_scaling
            min_scaling = torch.min(torch.abs(scaling), dim=-1)[0].mean()
            lambda_scale = self.GS_C(iteration, self.GS_config.opt.get('lambda_scale', 0.))
            loss += lambda_scale * min_scaling

            # regularization
            loss_reg = render_pkg["loss_reg"]
            for name, value in loss_reg.items():
                lbd = self.opt.get(f"lambda_{name}", 0.)
                lbd = self.GS_C(iteration, lbd)
                loss += lbd * value
            loss.backward()

            iter_end.record()
            torch.cuda.synchronize()

            with torch.no_grad():
                elapsed = iter_start.elapsed_time(iter_end)
                log_loss = {
                    'loss/l1_loss': loss_l1.item(),
                    'loss/ssim_loss': loss_dssim.item(),
                    'loss/perceptual_loss': loss_perceptual.item(),
                    'loss/mask_loss': loss_mask.item(),
                    'loss/loss_skinning': loss_skinning.item(),
                    'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                    'loss/cov_aiap_loss': loss_aiap_cov.item(),
                    'loss/min_scaling': min_scaling.item(),
                    'loss/total_loss': loss.item(),
                    'iter_time': elapsed,
                }
                log_loss.update({
                    'loss/loss_' + k: v for k, v in loss_reg.items()
                })
                wandb.log(log_loss)

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == self.opt.iterations:
                    progress_bar.close()

                # Log and save
                self.validation(iteration, self.testing_iterations, self.testing_interval, self.scene, self.evaluator,
                                (self.pipe, self.background))
                if (iteration in self.saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    self.scene.save(iteration)

                # Densification
                if iteration < self.opt.densify_until_iter and iteration > self.model.gaussian.delay:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.max_radii2D[visibility_filter] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(self.opt, self.scene, size_threshold)

                    if iteration % self.opt.opacity_reset_interval == 0 or (
                            self.dataset.white_background and iteration == self.opt.densify_from_iter):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.scene.optimize(iteration)

                if iteration in self.checkpoint_iterations:
                    self.scene.save_checkpoint(iteration)
