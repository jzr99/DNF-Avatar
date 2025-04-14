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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import pytorch3d.ops as ops


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
                                        rotation).permute(0, 2, 1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:, :3, :3] = RS
            trans[:, 3, :3] = center
            trans[:, 3, 3] = 1
            # H in the paper
            return trans

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.material_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    # def __init__(self, sh_degree: int):
    #     self.active_sh_degree = 0
    #     self.max_sh_degree = sh_degree
    #     self._xyz = torch.empty(0)
    #     self._features_dc = torch.empty(0)
    #     self._features_rest = torch.empty(0)
    #     self._scaling = torch.empty(0)
    #     self._rotation = torch.empty(0)
    #     self._opacity = torch.empty(0)
    #     self.max_radii2D = torch.empty(0)
    #     self.xyz_gradient_accum = torch.empty(0)
    #     self.denom = torch.empty(0)
    #     self.optimizer = None
    #     self.percent_dense = 0
    #     self.spatial_lr_scale = 0
    #     self.setup_functions()

    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # PBR properties
        self._normal = torch.empty(0)
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metallic = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.gs_type = "2dgs"

        self.albedo_bias = 0.03
        self.albedo_scale = 0.77
        self.roughness_bias = 0.09
        self.roughness_scale = 0.9

    def clone(self):
        cloned = GaussianModel(self.cfg)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      "_normal",
                      "_albedo",
                      "_roughness",
                      "_metallic"]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned


    # def capture(self):
    #     return (
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._features_dc,
    #         self._features_rest,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity,
    #         self.max_radii2D,
    #         self.xyz_gradient_accum,
    #         self.denom,
    #         self.optimizer.state_dict(),
    #         self.spatial_lr_scale,
    #     )

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._normal,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )



    # def restore(self, model_args, training_args):
    #     (self.active_sh_degree,
    #      self._xyz,
    #      self._features_dc,
    #      self._features_rest,
    #      self._scaling,
    #      self._rotation,
    #      self._opacity,
    #      self.max_radii2D,
    #      xyz_gradient_accum,
    #      denom,
    #      opt_dict,
    #      self.spatial_lr_scale) = model_args
    #     self.training_setup(training_args)
    #     self.xyz_gradient_accum = xyz_gradient_accum
    #     self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self._normal,
        self._albedo,
        self._roughness,
        self._metallic,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # .clamp(max=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_normal(self) -> torch.Tensor:
        # TODO change to min scale normal (maybe not change here, change at the usage part) (pbr_shading_gs in shader.py, and occ part)
        assert hasattr(self, 'rotation_precomp')
        # TODO we should consider the possible flipping of the normal (check the cuda code for how to do it)
        return F.normalize(self.rotation_precomp[:, :, -1], p=2, dim=-1)
        # return F.normalize(self._normal, p=2, dim=-1)

    @property
    def get_canonical_normal(self) -> torch.Tensor:
        rotation_matrix = build_rotation(self._rotation)
        # TODO we should consider the possible flipping of the normal (check the cuda code for how to do it)
        return F.normalize(rotation_matrix[:, :, -1], p=2, dim=-1)

    @property
    def get_albedo(self) -> torch.Tensor:
        return self.material_activation(self._albedo) * self.albedo_scale + self.albedo_bias

    @property
    def get_roughness(self) -> torch.Tensor:
        return self.material_activation(self._roughness) * self.roughness_scale + self.roughness_bias

    @property
    def get_metallic(self) -> torch.Tensor:
        return self.material_activation(self._metallic)

    def get_covariance(self, scaling_modifier=1):
        # TODO here the covariance is different from 3dgs (attention full aiap loss full_aiap_loss, maybe only use RS)
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # features[:, :3, 0] = fused_color
        # features[:, 3:, 1:] = 0.0

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        # init
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        normal = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        normal[..., 2] = 1.0
        albedo = torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        roughness = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        metallic = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {"params": [self._normal], "lr": training_args.rotation_lr, "name": "normal"},
            {"params": [self._albedo], "lr": training_args.opacity_lr, "name": "albedo"},
            {"params": [self._roughness], "lr": training_args.opacity_lr, "name": "roughness"},
            {"params": [self._metallic], "lr": training_args.opacity_lr, "name": "metallic"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._normal.shape[1]):
            l.append(f"normal_{i}")
        for i in range(self._albedo.shape[1]):
            l.append(f"albedo_{i}")
        l.append("roughness")
        l.append("metallic")
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, normal, albedo, roughness, metallic, scale, rotation), axis=1)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        normal = np.stack(
            (
                np.asarray(plydata.elements[0]["normal_0"]),
                np.asarray(plydata.elements[0]["normal_1"]),
                np.asarray(plydata.elements[0]["normal_2"]),
            ),
            axis=1,
        )
        albedo = np.stack(
            (
                np.asarray(plydata.elements[0]["albedo_0"]),
                np.asarray(plydata.elements[0]["albedo_1"]),
                np.asarray(plydata.elements[0]["albedo_2"]),
            ),
            axis=1,
        )
        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
        metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(
            torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._albedo = nn.Parameter(
            torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._roughness = nn.Parameter(
            torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._metallic = nn.Parameter(
            torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
    #                           new_rotation):
    #     d = {"xyz": new_xyz,
    #          "f_dc": new_features_dc,
    #          "f_rest": new_features_rest,
    #          "opacity": new_opacities,
    #          "scaling": new_scaling,
    #          "rotation": new_rotation}
    #
    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)
    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]
    #
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_albedo, new_roughness, new_metallic, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "normal": new_normal,
        "albedo": new_albedo,
        "roughness": new_roughness,
        "metallic": new_metallic,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._normal = optimizable_tensors["normal"]
        self._albedo = optimizable_tensors["albedo"]
        self._roughness = optimizable_tensors["roughness"]
        self._metallic = optimizable_tensors["metallic"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_normal, new_albedo,
                                   new_roughness, new_metallic, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_roughness = self._roughness[selected_pts_mask]
        new_metallic = self._metallic[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
        #                            new_rotation)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_albedo, new_roughness, new_metallic, new_scaling, new_rotation)

    def density_distillation(self, init_point, cfg):
        knn_ret = ops.knn_points(init_point.unsqueeze(0), self._xyz.unsqueeze(0), K=cfg.density_distillation_K, return_nn=False)
        distance_batch = knn_ret.dists[0] # (P1, K)
        index_batch = knn_ret.idx[0]  # (P1, K)

        print("Densification KNN mean distance : ", torch.mean(distance_batch))
        print("Densification KNN min distance : ", torch.min(distance_batch).values)
        print("Densification KNN max distance : ", torch.max(distance_batch).values)

        average_distance = torch.mean(distance_batch, dim=1)
        selected_pts_mask = torch.logical_and(average_distance > cfg.density_distillation_min, average_distance < cfg.density_distillation_max) # (P1)
        selected_pts_index = index_batch[selected_pts_mask] # (selected_p1, K)

        print("Number of points at density distilled : ", selected_pts_index.shape[0])
        print("Number of points at density distilled : ", selected_pts_index.shape[0])
        print("Number of points at density distilled : ", selected_pts_index.shape[0])

        new_xyz = init_point[selected_pts_mask] # (selected_p1, 3)
        new_features_dc = self._features_dc[selected_pts_index].mean(dim=1) # (selected_p1, D)
        new_features_rest = self._features_rest[selected_pts_index].mean(dim=1) # (selected_p1, D)
        new_opacities = self._opacity[selected_pts_index].mean(dim=1) # (selected_p1, 1)
        new_normal = self._normal[selected_pts_index].mean(dim=1) # (selected_p1, 3)
        new_albedo = self._albedo[selected_pts_index].mean(dim=1) # (selected_p1, 3)
        new_roughness = self._roughness[selected_pts_index].mean(dim=1) # (selected_p1, 1)
        new_metallic = self._metallic[selected_pts_index].mean(dim=1) # (selected_p1, 1)
        new_scaling = self._scaling[selected_pts_index].mean(dim=1) # (selected_p1, 2)
        new_rotation = self._rotation[selected_pts_index].mean(dim=1) # (selected_p1, 4)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_albedo,
                                   new_roughness, new_metallic, new_scaling, new_rotation)
        torch.cuda.empty_cache()

        # Extract points that satisfy the gradient condition

        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling,
        #                                                 dim=1).values <= self.percent_dense * scene_extent)
        #
        # new_xyz = self._xyz[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask]
        # new_features_rest = self._features_rest[selected_pts_mask]
        # new_opacities = self._opacity[selected_pts_mask]
        # new_normal = self._normal[selected_pts_mask]
        # new_albedo = self._albedo[selected_pts_mask]
        # new_roughness = self._roughness[selected_pts_mask]
        # new_metallic = self._metallic[selected_pts_mask]
        # new_scaling = self._scaling[selected_pts_mask]
        # new_rotation = self._rotation[selected_pts_mask]
        #
        # # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
        # #                            new_rotation)
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_normal, new_albedo, new_roughness, new_metallic, new_scaling, new_rotation)

    def density_prune_distillation(self, init_point, K, outlier_threshold):
        knn_ret = ops.knn_points(self._xyz.unsqueeze(0), init_point.unsqueeze(0), K=K,
                                 return_nn=False)
        distance_batch = knn_ret.dists[0]  # (P1, K)
        index_batch = knn_ret.idx[0]  # (P1, K)

        print("Densification KNN mean distance : ", torch.mean(distance_batch))
        print("Densification KNN min distance : ", torch.min(distance_batch).values)
        print("Densification KNN max distance : ", torch.max(distance_batch).values)

        average_distance = torch.mean(distance_batch, dim=1)

        outlier_mask = average_distance > outlier_threshold
        prune_mask = outlier_mask

        print("Number of points pruned during distillation : ", outlier_mask.sum())
        print("Number of points pruned during distillation : ", outlier_mask.sum())
        print("Number of points pruned during distillation : ", outlier_mask.sum())

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()



    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0
    #
    #     self.densify_and_clone(grads, max_grad, extent)
    #     self.densify_and_split(grads, max_grad, extent)
    #
    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)
    #
    #     torch.cuda.empty_cache()

    def densify_and_prune(self, opt, scene, max_screen_size):
        extent = scene.cameras_extent

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # TODO check viewspace_point_tensor.grad[update_filter,:2]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def prune_nan(self):
        nan_mask = self._xyz.isnan().any(dim=-1)
        if self._xyz.grad is not None:
            nan_grad_mask = self._xyz.grad.isnan().any(dim=-1)
            nan_mask = torch.logical_or(nan_mask, nan_grad_mask)
        self.prune_points(nan_mask)
        torch.cuda.empty_cache()


    def prune_large_condition_number(self):
        # min_opacity = 0.01
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        scale = self.get_scaling
        large_condition_mask = (scale[:, 0] / scale[:, 1] > 5) | (scale[:, 1] / scale[:, 0] > 5)
        # prune_mask = torch.logical_or(prune_mask, large_condition_mask)
        prune_mask = large_condition_mask

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_small_opacity(self):
        min_opacity = 0.01
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()