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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixShift

class Camera:
    def __init__(self, camera=None, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs

        if 'valid_msk' in kwargs.keys() and kwargs['valid_msk'] is not None:
            self.data['valid_msk'] = self.valid_msk.to(self.data_device)
        else:
            self.data['valid_msk'] = None

        if 'hdri' in kwargs.keys() and self.hdri is not None:
            self.data['hdri'] = self.hdri.to(self.data_device)
        else:
            self.data['hdri'] = None

        if 'gt_albedo' in kwargs.keys() and self.gt_albedo is not None:
            self.data['gt_albedo'] = self.gt_albedo.to(self.data_device)
            self.data['gt_normal'] = self.gt_normal.to(self.data_device)
        else:
            self.data['gt_albedo'] = None
            self.data['gt_normal'] = None

        if self.normal_img is not None:
            self.data['normal_img'] = self.normal_img.to(self.data_device) # [3, H, W]
            self.data['depth_value'] = self.depth_value.to(self.data_device) # [H, W]
            self.data['albedo_img'] = self.albedo_img.to(self.data_device) # [3, H, W]

        if self.roughness_img is not None:
            self.data['roughness_img'] = self.roughness_img.to(self.data_device) # [3, H, W]
            self.data['metallic_img'] = self.metallic_img.to(self.data_device) # [3, H, W]

        self.data['w2c_opencv'] = self.w2c_opencv.to(self.data_device)

        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0

        self.data['original_image'] = self.image.clamp(0.0, 1.0).to(self.data_device)
        self.data['image_width'] = self.original_image.shape[2]
        self.data['image_height'] = self.original_image.shape[1]
        self.data['original_mask'] = self.mask.float().to(self.data_device)

        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        self.data['world_view_transform'] = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()

        self.data['shift_projection_matrix'] = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, focal_x=self.K[0,0], focal_y=self.K[1,1], cx=self.K[0,2], cy=self.K[1,2], width=self.image_width, height=self.image_height, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['shift_full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.shift_projection_matrix.unsqueeze(0))).squeeze(0)

        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3]

        self.data['rots'] = self.rots.to(self.data_device)
        self.data['Jtrs'] = self.Jtrs.to(self.data_device)
        self.data['bone_transforms'] = self.bone_transforms.to(self.data_device)


    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = Camera(camera=self)
        return new_cam

    def merge(self, cam):
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()
