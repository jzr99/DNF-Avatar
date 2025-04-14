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

import os
import torch
import torch.nn as nn
from models_GS import GaussianConverter
# from scene.gaussian_model import GaussianModel
from dataset import load_dataset
import numpy as np
from pbr import CubemapLight, get_brdf_lut, pbr_shading


class Scene:

    # gaussians : GaussianModel

    def __init__(self, cfg, gaussians, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians

        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError
        self.distill_dataset = None

        self.cameras_extent = self.metadata['cameras_extent']

        self.init_pcd = self.test_dataset.readPointCloud()
        self.init_mode = cfg.dataset.init_mode

        self.gaussians.create_from_pcd(self.init_pcd, spatial_lr_scale=self.cameras_extent)

        self.init_point_cloud = torch.tensor(np.asarray(self.init_pcd.points)).float().cuda()

        self.converter = GaussianConverter(cfg, self.metadata).cuda()
        base_res = cfg.get('base_res', 256)
        self.cubemap = CubemapLight(base_res=base_res).cuda()
        # aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
        # irradiance_volumes = IrradianceVolumes(aabb=aabb).cuda()
        # irradiance_volumes.train()

        self.distill_adapt_metallic_layer = nn.Conv1d(1, 1, 1, stride=1, device='cuda')
        self.distill_adapt_roughness_layer = nn.Conv1d(1, 1, 1, stride=1, device='cuda')
        if cfg.dataset.name == 'people_snapshot' or cfg.dataset.name == 'animation':
            self.albedo_adapt_params = nn.Parameter(torch.ones(3,1, requires_grad=True).cuda())
        else:
            self.albedo_adapt_params = nn.Parameter(torch.tensor(0.75 * torch.ones(3,1)).cuda())
        param_groups = [
            # {
            #     "name": "irradiance_volumes",
            #     "params": irradiance_volumes.parameters(),
            #     "lr": opt.opacity_lr,
            # },
            {"name": "cubemap", "params": self.cubemap.parameters(), "lr": self.cfg.opt.opacity_lr},
            {"name": "distill_adapt_metallic_layer", "params": self.distill_adapt_metallic_layer.parameters(), "lr": self.cfg.opt.opacity_lr},
            {"name": "distill_adapt_roughness_layer", "params": self.distill_adapt_roughness_layer.parameters(), "lr": self.cfg.opt.opacity_lr},
            {"name": "albedo_adapt_params", "params": self.albedo_adapt_params, "lr": self.cfg.opt.opacity_lr},
        ]
        self.light_optimizer = torch.optim.Adam(param_groups, lr=self.cfg.opt.opacity_lr)

    def train(self):
        self.converter.train()
        self.cubemap.train()

    def eval(self):
        self.converter.eval()
        self.cubemap.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()
        # if iteration > self.cfg.model.pbr_iteration:
        self.light_optimizer.step()
        self.light_optimizer.zero_grad(set_to_none=True)
        self.cubemap.clamp_(min=0.0)


    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True, compute_color=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss, compute_color)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        print("self.albedo_adapt_params: ", self.albedo_adapt_params)
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    self.cubemap.state_dict(),
                    self.light_optimizer.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, cubemap_params, light_optimizer_sd, first_iter) = torch.load(path)
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd, strict=False)
        try:
            # import pdb;pdb.set_trace()
            self.cubemap.load_state_dict(cubemap_params) # hack here
        except:
            print("cubemap load failed, due to the change of the base_res")
        # load optimizer
        self.converter.optimizer.load_state_dict(converter_opt_sd)
        self.converter.scheduler.load_state_dict(converter_scd_sd)
        self.light_optimizer.load_state_dict(light_optimizer_sd)