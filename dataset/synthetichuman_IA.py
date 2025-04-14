import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import pickle as pkl
from utils.dataset_utils import get_02v_bone_transforms, get_apose_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh
from scripts.body_model.body_model import BodyModel

import json


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

class SynthetichumanDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.subject = cfg.subject
        self.train_frames = cfg.train_frames
        self.val_frames = cfg.val_frames
        self.white_bg = cfg.white_background
        # self.split = cfg.split

        self.rest_pose = cfg.get('rest_pose', 'star_pose')

        if split == 'train':
            frames = self.train_frames
        elif split == 'val':
            frames = self.val_frames
        elif split == 'test':
            frames = self.cfg.test_frames[self.cfg.test_mode]
        elif split == 'predict':
            raise NotImplementedError
            # frames = self.cfg.predict_frames
        else:
            raise ValueError
        root = os.path.join(self.root_dir, self.subject)
        with open(os.path.join(root, "cameras.json"), "r") as f:
            cameras = json.load(f)

        self.all_cam_names = cameras["all_cam_names"]
        cam_name = "00"
        camera = cameras[cam_name]

        if split == "test":
            with open(os.path.join(root, "hdri_files.json"), "r") as f:
                hdri_files = json.load(f)

            self.hdri_files = [os.path.join(self.root_dir, "hdri", f) for f in hdri_files]

        self.K = np.array(camera["intrinsic"], dtype=np.float32)
        self.c2w = np.linalg.inv(np.array(camera["extrinsic"], dtype=np.float32))
        try:
            self.D = np.array(camera["distortion"], dtype=np.float32)
        except:
            self.D = np.zeros(4)
        self.w2c_opencv = np.array(camera["extrinsic"], dtype=np.float32)
        self.R = self.w2c_opencv[:3, :3]
        self.T = self.w2c_opencv[:3, 3:]
        self.H = camera["height"]
        self.W = camera["width"]

        self.h, self.w = cfg.img_hw
        # import pdb; pdb.set_trace()

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        # gender = 'female' if 'female' in self.subject else 'male'
        gender = cfg.gender

        start_frame, end_frame, sampling_rate = frames

        cam_idx = 0
        # cam_name = '1'

        subject_dir = root
        if split == 'predict':
            raise NotImplementedError
            # predict_seqs = ['rotating_models',
            #                 'gLO_sBM_cAll_d14_mLO1_ch05_view1']
            # predict_seq = self.cfg.get('predict_seq', 0)
            # predict_seq = predict_seqs[predict_seq]
            # model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
            # self.model_files = model_files
            # frames = list(reversed(range(-len(model_files), 0)))
            # if end_frame == 0:
            #     end_frame = len(model_files)
            # frame_slice = slice(start_frame, end_frame, sampling_rate)
            # model_files = model_files[frame_slice]
            # frames = frames[frame_slice]
        else:

            # if os.path.exists(os.path.join(root, f"poses/anim_nerf_{split}.npz")):
            #     cached_path = os.path.join(root, f"poses/anim_nerf_{split}.npz")
            # elif os.path.exists(os.path.join(root, f"poses/{split}.npz")):
            #     cached_path = os.path.join(root, f"poses/{split}.npz")
            # else:
            #     cached_path = None
            # print(f"[{split}] Loading from", cached_path)
            # self.smpl_params = load_smpl_param(cached_path)
            print(f"[{split}] No optimized smpl found.")
            self.smpl_params = load_smpl_param(os.path.join(root, "poses.npz"))
            for k, v in self.smpl_params.items():
                if k != "betas":
                    self.smpl_params[k] = v[start_frame:end_frame:sampling_rate]

            batch_number = self.smpl_params['global_orient'].shape[0]
            body_model = BodyModel(bm_path=f'body_models/smpl/{gender}/model.pkl', num_betas=10, batch_size=batch_number)
            # import pdb;pdb.set_trace()
            body = body_model(root_orient=torch.from_numpy(self.smpl_params['global_orient']), pose_body=torch.from_numpy(self.smpl_params['body_pose'][:, :63]),
                              betas=torch.from_numpy(self.smpl_params['betas']).repeat((batch_number,1)), trans=torch.from_numpy(self.smpl_params['transl']), pose_hand=torch.from_numpy(self.smpl_params['body_pose'][:, 63:]))

            bone_transforms = body.bone_transforms.detach().cpu().numpy()
            # Jtr_posed = body.Jtr.detach().cpu().numpy()
            body_model_m = BodyModel(bm_path=f'body_models/smpl/{gender}/model.pkl', num_betas=10,
                                   batch_size=1)
            body_minimal = body_model_m(betas=torch.from_numpy(self.smpl_params['betas']))
            minimal_shape = body_minimal.v.detach().cpu().numpy()[0]
            # minimal_shape = body_minimal.v.detach().cpu().numpy()[0]


            # check here we assume the parameter is skipped
            frames = list(range(start_frame, end_frame, sampling_rate))
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            # model_files = [os.path.join(subject_dir, f'animnerf_models/{frame:06d}.npz') for frame in frames]
            # self.model_files = model_files


        self.data = []
        if split == 'test':
            img_dir = os.path.join(subject_dir, cam_name, 'images_relit')
        else:
            img_dir = os.path.join(subject_dir, cam_name, 'images')
        mask_dir = os.path.join(subject_dir, cam_name, 'masks')
        albedo_dir = os.path.join(subject_dir, cam_name, 'albedos_png')
        normal_dir = os.path.join(subject_dir, cam_name, 'normals_png')
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.npy')))
        albedo_lists = sorted(glob.glob(os.path.join(albedo_dir, '*.png')))
        normal_lists = sorted(glob.glob(os.path.join(normal_dir, '*.png')))
        img_files = img_files[frame_slice]
        mask_files = mask_files[frame_slice]
        albedo_lists = albedo_lists[frame_slice]
        normal_lists = normal_lists[frame_slice]
        if split == 'train':
            distill_dir = cfg.distill_data_dir
            distill_normal_files = sorted(glob.glob(os.path.join(distill_dir, '*-normal.png')))
            distill_depth_files = sorted(glob.glob(os.path.join(distill_dir, '*-depth.npy')))
            distill_albedo_files = sorted(glob.glob(os.path.join(distill_dir, '*_albedo.png')))
            assert len(distill_normal_files) == len(distill_depth_files) == len(distill_albedo_files)
            assert len(distill_normal_files) == len(img_files) == len(mask_files)

        assert len(self.smpl_params['body_pose']) == len(img_files) == len(mask_files)

        if split == 'predict':
            for d_idx, f_idx in enumerate(frames):
                model_file = {
                    'bone_transforms': bone_transforms[d_idx],
                    'root_orient': self.smpl_params["global_orient"][d_idx],
                    'pose_body': self.smpl_params["body_pose"][d_idx, :63],
                    'pose_hand': self.smpl_params["body_pose"][d_idx, 63:],
                    'trans': self.smpl_params["transl"][d_idx],
                }
                # model_file = model_files[d_idx]
                # get dummy gt...
                img_file = img_files[0]
                mask_file = mask_files[0]

                self.data.append({
                    'cam_idx': cam_idx,
                    'cam_name': cam_name,
                    'data_idx': d_idx,
                    'frame_idx': f_idx,
                    'img_file': img_file,
                    'mask_file': mask_file,
                    'model_file': model_file,
                    })
        else:
            for d_idx, f_idx in enumerate(frames):
                img_file = img_files[d_idx]
                mask_file = mask_files[d_idx]
                gt_normal_file = normal_lists[d_idx]
                gt_albedo_file = albedo_lists[d_idx]
                # model_file = model_files[d_idx]
                model_file = {
                    'bone_transforms': bone_transforms[d_idx],
                    'root_orient': self.smpl_params["global_orient"][d_idx],
                    'pose_body': self.smpl_params["body_pose"][d_idx, :63],
                    'pose_hand': self.smpl_params["body_pose"][d_idx, 63:],
                    'trans': self.smpl_params["transl"][d_idx],
                }
                if split == 'train':
                    normal_file = distill_normal_files[d_idx]
                    depth_file = distill_depth_files[d_idx]
                    albedo_file = distill_albedo_files[d_idx]
                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                        'normal_file': normal_file,
                        'depth_file': depth_file,
                        'albedo_file': albedo_file,
                        'gt_normal_file': gt_normal_file,
                        'gt_albedo_file': gt_albedo_file,
                    })
                else:
                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                        'gt_normal_file': gt_normal_file,
                        'gt_albedo_file': gt_albedo_file,
                    })


        self.frames = frames
        # self.model_files_list = model_files

        self.get_metadata(minimal_shape, gender, number_frames=len(frames))

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

    @staticmethod
    def get_KRTD(camera):
        K = np.zeros([3, 3], dtype=np.float32)
        K[0, 0] = camera['camera_f'][0]
        K[1, 1] = camera['camera_f'][1]
        K[:2, 2] = camera['camera_c']
        K[2, 2] = 1
        R = np.eye(3, dtype=np.float32)
        T = np.zeros([3, 1], dtype=np.float32)
        D = camera['camera_k']

        return K, R, T, D

    def get_metadata(self, minimal_shape, gender, number_frames):
        # data_paths = self.model_files
        # data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(minimal_shape, gender)
        if self.split != 'train':
            self.metadata = cano_data
            return
        # TODO a big problem here, do not know if a second selection is a bug or not, need to check with zhiyin
        start, end, step = self.train_frames
        frames = list(range(number_frames))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }
        self.metadata.update(cano_data)
        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())

    def get_cano_smpl_verts(self, minimal_shape, gender):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        # model_dict = np.load(data_path)
        # gender = 'female' if 'female' in self.subject else 'male'

        # 3D models and points
        # minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        if self.rest_pose == 'star_pose':
            bone_transforms_02v = get_02v_bone_transforms(Jtr)
        elif self.rest_pose == 'a_pose':
            bone_transforms_02v = get_apose_bone_transforms(Jtr)
        else:
            raise NotImplementedError

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        # load all smpl parameters of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, frame in enumerate(self.frames):
            # model_dict = np.load(model_file)
            if idx == 0:
                smpl_data['betas'] = self.smpl_params['betas'].astype(np.float32)

            smpl_data['frames'].append(frame)
            smpl_data['root_orient'].append(self.smpl_params['global_orient'][idx].astype(np.float32))
            smpl_data['pose_body'].append(self.smpl_params['body_pose'][idx, :63].astype(np.float32))
            smpl_data['pose_hand'].append(self.smpl_params['body_pose'][idx, 63:].astype(np.float32))
            smpl_data['trans'].append(self.smpl_params['transl'][idx].astype(np.float32))

        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx):
        data_dict = self.data[idx]
        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']
        gt_normal_file = data_dict.get('gt_normal_file', None)
        gt_albedo_file = data_dict.get('gt_albedo_file', None)

        K = self.K.copy()
        dist = self.D.copy()
        R = self.R.copy()
        T = self.T.copy()

        # note that in ZJUMoCap the camera center does not align perfectly in the intrinsic
        # here we try to offset it by modifying the extrinsic...

        # M = np.eye(3)
        # M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        # M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        # K[0, 2] = self.W / 2
        # K[1, 2] = self.H / 2
        # R = M @ R
        # T = M @ T
        # TODO check if we need to transpose R (change from opencv camera to OpenGL camera)
        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = np.load(mask_file)
        # mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        # todo check if the two GT images are aligned (IA & GS)
        gt_albedo = cv2.cvtColor(cv2.imread(gt_albedo_file), cv2.COLOR_BGR2RGB)
        gt_normal = cv2.cvtColor(cv2.imread(gt_normal_file), cv2.COLOR_BGR2RGB)

        image = cv2.undistort(image, K, dist, None)
        mask = cv2.undistort(mask, K, dist, None)
        gt_albedo = cv2.undistort(gt_albedo, K, dist, None)
        gt_normal = cv2.undistort(gt_normal, K, dist, None)

        normal_img = None
        depth_value = None
        albedo_img = None
        if self.split == 'train' and data_dict.get('normal_file', None) is not None:
            normal_file = data_dict['normal_file']
            depth_file = data_dict['depth_file']
            albedo_file = data_dict['albedo_file']
            normal_img = cv2.cvtColor(cv2.imread(normal_file), cv2.COLOR_BGR2RGB)
            albedo_img = cv2.cvtColor(cv2.imread(albedo_file), cv2.COLOR_BGR2RGB)
            # normal_img = cv2.undistort(normal_img, K, dist, None)
            normal_img = normal_img / 255.
            normal_img = torch.from_numpy(normal_img).permute(2, 0, 1).float()
            albedo_img = albedo_img / 255.
            albedo_img = torch.from_numpy(albedo_img).permute(2, 0, 1).float()
            depth_value = np.load(depth_file)
            depth_value = torch.from_numpy(depth_value).float()

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        gt_albedo = cv2.resize(gt_albedo, (self.w, self.h), interpolation=interpolation)
        gt_normal = cv2.resize(gt_normal, (self.w, self.h), interpolation=interpolation)

        # Also compute foreground region mask for evaluation purpose
        # we dilate the foreground region mask to account for potential
        # pose misalignment for novel pose relighting evaluation
        dilate_kernel = np.ones((50, 50), np.uint8)
        msk_dilate = cv2.dilate(mask.astype(np.uint8), dilate_kernel, iterations=1)
        x, y, w, h = cv2.boundingRect(msk_dilate)
        valid_msk = np.zeros(msk_dilate.shape, dtype=bool)
        valid_msk[y: y + h, x: x + w] = True

        gt_albedo = (gt_albedo[..., :3] / 255).astype(np.float32)
        gt_normal = ((gt_normal[..., :3] / 255).astype(np.float32) - 0.5) * 2


        # mask = mask != 0
        mask = (mask > 0.5)
        image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        valid_msk = torch.from_numpy(valid_msk).unsqueeze(0).bool()
        gt_albedo = torch.from_numpy(gt_albedo).permute(2, 0, 1).float()
        gt_normal = torch.from_numpy(gt_normal).permute(2, 0, 1).float()

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']
        gender = self.metadata['gender']

        # model_dict = np.load(model_file)
        model_dict = model_file
        n_smpl_points = minimal_shape.shape[0]
        trans = model_dict['trans'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        # Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot_full = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]
        Jtr = self.metadata['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans  # add global offset

        hdri = None
        if self.split == "test":
            hdri = cv2.cvtColor(
                cv2.imread(
                    self.hdri_files[idx], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
                ),
                cv2.COLOR_BGR2RGB,
            )
            # hdri = cv2.resize(hdri, dsize=None)
            hdri = torch.from_numpy(hdri.astype(np.float32))

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
            normal_img=normal_img,
            depth_value=depth_value,
            albedo_img = albedo_img,
            valid_msk=valid_msk,
            hdri=hdri,
            gt_albedo=gt_albedo,
            gt_normal=gt_normal,
            w2c_opencv=torch.from_numpy(self.w2c_opencv),
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    # def readPointCloud(self,):
    #     if self.cfg.get('random_init', False):
    #         ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')
    #
    #         aabb = self.metadata['aabb']
    #         coord_min = aabb.coord_min.unsqueeze(0).numpy()
    #         coord_max = aabb.coord_max.unsqueeze(0).numpy()
    #         n_points = 50_000
    #
    #         xyz_norm = np.random.rand(n_points, 3)
    #         xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
    #         rgb = np.ones_like(xyz) * 255
    #         storePly(ply_path, xyz, rgb)
    #
    #         pcd = fetchPly(ply_path)
    #     else:
    #         ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
    #         try:
    #             pcd = fetchPly(ply_path)
    #         except:
    #             verts = self.metadata['smpl_verts']
    #             faces = self.faces
    #             mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    #             n_points = 50_000
    #
    #             xyz = mesh.sample(n_points)
    #             rgb = np.ones_like(xyz) * 255
    #             storePly(ply_path, xyz, rgb)
    #
    #             pcd = fetchPly(ply_path)
    #
    #     return pcd

    def readPointCloud(self,):
        if self.cfg.init_mode == 'random':
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        elif self.cfg.init_mode == 'smpl':
            ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
            # try:
            #     pcd = fetchPly(ply_path)
            # except:
            verts = self.metadata['smpl_verts']
            faces = self.faces
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            n_points = 50_000

            xyz = mesh.sample(n_points)
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)

        elif self.cfg.init_mode == 'IA':

            filename = self.cfg.obj_path.split('/')[-1].split('.')[0]
            ply_path = os.path.join(self.root_dir, self.subject, filename + '_cano_IA.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                obj_mesh = trimesh.load_mesh(self.cfg.obj_path)
                # covert to trimesh
                xyz = obj_mesh.sample(50_000)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)
                pcd = fetchPly(ply_path)
        else:
            raise NotImplementedError

        return pcd

