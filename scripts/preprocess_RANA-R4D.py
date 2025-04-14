import os
import sys
import torch

import cv2
import glob
import re
import json
import shutil
import argparse

import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.spatial.transform import Rotation as Rotation

# body models
from models.deformers.smplx import SMPL
from models.utils import get_perspective
from scripts.easymocap.smplmodel import SMPLlayer as SMPLlayerEM

from utils.smpl_renderer import Renderer

parser = argparse.ArgumentParser(description="Preprocessing for RANA.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains raw RANA data."
)
parser.add_argument(
    "--split",
    type=str,
    help="Split to process.",
    choices=["train_p1", "train_p2_p3", "test"],
)
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="subject_01", help="Sequence to process."
)
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, args.split, seq_name)
    out_dir = os.path.join(args.out_dir, args.split, seq_name)

    os.makedirs(out_dir, exist_ok=True)
    shape = []
    global_orient = []
    body_pose = []
    transl = []

    device = torch.device("cuda")

    if args.visualize:
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(vis_dir, exist_ok=True)

    img_out_dir = os.path.join(out_dir, "image")
    os.makedirs(img_out_dir, exist_ok=True)
    albedo_out_dir = os.path.join(out_dir, "albedo")
    os.makedirs(albedo_out_dir, exist_ok=True)
    normal_out_dir = os.path.join(out_dir, "normal")
    os.makedirs(normal_out_dir, exist_ok=True)
    mask_out_dir = os.path.join(out_dir, "mask")
    os.makedirs(mask_out_dir, exist_ok=True)
    vertices_out_dir = os.path.join(out_dir, "vertices")
    os.makedirs(vertices_out_dir, exist_ok=True)

    img_pattern = re.compile(r"frame_(\d{6})\.png$")
    img_files = sorted(glob.glob(os.path.join(data_dir, "frame_*.png")))
    img_files = [f for f in img_files if img_pattern.match(os.path.basename(f))]

    # Load gender information
    base_name = os.path.basename(img_files[0]).split(".")[0]
    json_file = os.path.join(data_dir, base_name + ".json")
    with open(json_file, 'r') as f:
        annots = json.load(f)

    smpl_data = annots['skeleton_0']['smpl_data']
    global_scale = np.array(smpl_data["scale"], dtype=np.float32)
    # Print global_scale to stderr
    print("Global scale of {}: {}".format(seq_name, global_scale), file=sys.stderr)

    # Standard SMPL model
    gender = np.array(annots['skeleton_0']['smpl_data']['gender']).tolist()

    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    # EasyMocap SMPL model
    body_model_em = SMPLlayerEM(
        "./data/SMPLX/smpl",
        model_type="smpl",
        gender=gender,
        device=device,
        regressor_path=os.path.join("data/smplh", "J_regressor_body25_smplh.txt"),
    )

    if args.split == "test":
        # Make directory for HDR maps
        hdri_dir = os.path.join(args.out_dir, "hdri")
        os.makedirs(hdri_dir, exist_ok=True)
        hdri_files = []

    for img_idx, img_file in enumerate(img_files):
        print("Processing: {}".format(img_file))

        base_name = os.path.basename(img_file).split(".")[0]
        json_file = os.path.join(data_dir, base_name + ".json")
        with open(json_file, "r") as f:
            annots = json.load(f)

        if args.split == "test":
            assert "bg_file" in annots.keys()
            assert "yaw" in annots['camera']
            assert "fov" in annots['camera']
            # Download and save HDR map
            url = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/{}".format(
                annots["bg_file"]
            )
            hdri_file = os.path.join(hdri_dir, os.path.basename(url))
            if not os.path.exists(hdri_file):
                os.system("wget {} -P {}".format(url, hdri_dir))

            hdri_files.append(os.path.basename(hdri_file))

            # # Load HDR image and get height/width
            # hdr_img = cv2.imread(hdri_file, cv2.IMREAD_COLOR)
            # hdr_height, hdr_width = hdr_img.shape[:2]

            yaw = annots['camera']['yaw']
            assert yaw == 0
            theta = -270 - yaw
            intrinsic, R = get_perspective(
                fov=np.rad2deg(annots['camera']['fov']),
                theta=theta,
                phi=0,
                height=720,
                width=1280,
            )
        else:
            # Intrinsics
            intrinsic = np.array(annots['skeleton_0']['smpl_data']['K'])
            # Extrinsics
            R = np.eye(3, dtype=np.float32)

        T = np.array([0, 0, 0], dtype=np.float32).reshape(3, 1)
        extrinsic = np.block([[R, T], [0, 0, 0, 1]])

        smpl_data = annots['skeleton_0']['smpl_data']
        # verts_smpl = np.array(smpl_data['vertices']) / 1000.0
        smpl_pose = np.array(smpl_data["pose"], dtype=np.float32).reshape(1, -1)
        smpl_pose[:, 57:] = 0.0    # set hand pose to zero
        smpl_betas = np.array(smpl_data["betas"], dtype=np.float32).reshape(1, -1)
        smpl_global_orient = np.array(
            smpl_data["global_orient"], dtype=np.float32
        ).reshape(1, -1)
        global_trans = np.array(
            smpl_data["global_trans"], dtype=np.float32
        ).reshape(3, 1)
        global_scale = np.array(smpl_data["scale"], dtype=np.float32)

        poses_torch = torch.from_numpy(smpl_pose).cuda()
        poses_em_torch = torch.cat([torch.zeros_like(poses_torch[..., :3]), poses_torch], dim=-1)
        betas_torch = torch.from_numpy(smpl_betas).cuda()
        global_orient_torch = torch.from_numpy(smpl_global_orient).cuda()

        out = body_model_smpl(
            betas=betas_torch, body_pose=poses_torch, global_orient=global_orient_torch
        )
        smpl_p3d = out.joints
        smpl_root = smpl_p3d[:, :1]
        # verts_rel = (out.vertices - smpl_root)[0].detach().cpu().numpy()
        # verts_smpl = (verts_rel * global_scale) + global_trans

        smpl_transl = (
            -smpl_root[0].detach().cpu().numpy()
            + global_trans.reshape(1, -1) / global_scale
        )
        transl_torch = torch.from_numpy(smpl_transl.reshape(1, -1)).cuda()

        # Record SMPL parameters
        if img_idx == 0:
            shape = smpl_betas.copy()
        else:
            assert (shape.flatten() == smpl_betas.flatten()).all()

        # SMPL mesh via standard SMPL
        smpl_outputs = body_model_smpl(
            betas=betas_torch,
            body_pose=poses_torch,
            global_orient=global_orient_torch,
            transl=transl_torch,
        )
        verts_smpl = smpl_outputs.vertices.detach().cpu().numpy()[0]
        # SMPL mesh via EasyMocap SMPL
        verts_em = (
            body_model_em(
                poses=poses_em_torch,
                shapes=betas_torch,
                Rh=global_orient_torch,
                Th=transl_torch,
                return_verts=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        # Compute new translation
        smpl_transl = smpl_transl + (verts_smpl - verts_em).mean(axis=0, keepdims=True)

        if args.split == "test":
            smpl_global_orient = (
                R.T @ Rotation.from_rotvec(smpl_global_orient).as_matrix()
            )
            global_orient_torch = torch.from_numpy(smpl_global_orient).cuda().float()
            smpl_transl = smpl_transl @ R
            smpl_global_orient = (
                Rotation.from_matrix(smpl_global_orient).as_rotvec().astype(np.float32)
            )

        transl_em_torch = torch.from_numpy(smpl_transl.reshape(1, -1)).cuda()

        global_orient.append(smpl_global_orient)
        body_pose.append(smpl_pose)
        transl.append(smpl_transl)

        # Visualize SMPL mesh
        if args.visualize:
            # Re-compute SMPL mesh with new translation
            verts_smpl = (
                body_model_em(
                    poses=poses_em_torch,
                    shapes=betas_torch,
                    Rh=global_orient_torch,
                    Th=transl_em_torch,
                    return_verts=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            assert verts_smpl.shape == (6890, 3)

            # Save SMPL vertices
            np.save(
                os.path.join(vertices_out_dir, "verts_{:04d}.npy".format(img_idx)),
                verts_smpl,
            )

            img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
            renderer = Renderer(
                height=img.shape[0],
                width=img.shape[1],
                faces=body_model_smpl.faces,
            )

            render_cameras = {
                "K": [intrinsic],
                "R": [np.array(R, dtype=np.float32)],
                "T": [np.array(T, dtype=np.float32)],
            }

            render_data = {
                0: {
                    "name": "SMPL",
                    "vertices": verts_smpl,
                    "faces": body_model_smpl.faces,
                    "vid": 2,
                }
            }

            images = [img]
            smpl_image = renderer.render(
                render_data,
                render_cameras,
                images,
                use_white=False,
                add_back=True,
            )[0]

            cv2.imwrite(
                os.path.join(vis_dir, "{}.jpg".format(base_name)),
                cv2.cvtColor(smpl_image, cv2.COLOR_RGB2BGR),
            )

            del renderer

        shutil.copy(
            os.path.join(img_file),
            os.path.join(img_out_dir, "image_{:04d}.png".format(img_idx)),
        )
        albedo_file = os.path.join(data_dir, base_name + "_albedo.png")
        shutil.copy(
            os.path.join(albedo_file),
            os.path.join(albedo_out_dir, "albedo_{:04d}.png".format(img_idx)),
        )
        normal_file = os.path.join(data_dir, base_name + "_normals.png")
        shutil.copy(
            os.path.join(normal_file),
            os.path.join(normal_out_dir, "normal_{:04d}.png".format(img_idx)),
        )
        mask_file = os.path.join(data_dir, base_name + "_semantic.png")
        rgba = Image.open(mask_file)
        rgba = TF.to_tensor(rgba).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
        h, w = rgba.shape[:2]
        mask = (rgba[..., -1:] > 0.5).byte().numpy().repeat(3, axis=-1)
        cv2.imwrite(os.path.join(mask_out_dir, "mask_{:04d}.png".format(img_idx)), mask)

        # Record camera parameters for this frame
        if img_idx == 0:
            cam_params = {
                "intrinsic": intrinsic.tolist(),
                "extrinsic": extrinsic.tolist(),
                "distortion": [0, 0, 0, 0],
                "height": h,
                "width": w,
            }
        else:
            assert cam_params["intrinsic"] == intrinsic.tolist()
            assert cam_params["extrinsic"] == extrinsic.tolist()
            assert cam_params["distortion"] == [0, 0, 0, 0]
            assert cam_params["height"] == h
            assert cam_params["width"] == w

    with open(os.path.join(out_dir, "cameras.json"), "w") as f:
        json.dump(cam_params, f)
    out_filename = os.path.join(out_dir, "params.npy")
    # Save easymocap SMPL parameters
    params = {
        "gender": gender,
        "beta": shape,
        "pose": np.concatenate(
            [np.concatenate(global_orient, axis=0), np.concatenate(body_pose, axis=0)],
            axis=-1,
        ),
        "transl": np.concatenate(transl, axis=0),
    }
    np.save(
        out_filename,
        params,
    )
    out_filename = os.path.join(out_dir, "hdri_files.json")
    with open(out_filename, "w") as f:
        json.dump(hdri_files, f)
