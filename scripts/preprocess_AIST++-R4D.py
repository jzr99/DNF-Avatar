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

parser = argparse.ArgumentParser(description="Preprocessing for AIST++ for R4D.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains AIST++ data."
)
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="male-3-casual", help="Sequence from which to load body shape."
)
parser.add_argument("--skip", type=int, default=2, help="Skip every n frames.")
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir)
    out_dir = os.path.join(args.out_dir, seq_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")

    if args.visualize:
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(vis_dir, exist_ok=True)

    vertices_out_dir = os.path.join(out_dir, "vertices")
    os.makedirs(vertices_out_dir, exist_ok=True)

    # Load SMPL shape
    shape_file = f"./load/peoplesnapshot/{args.seqname}/poses/anim_nerf_train.npz"
    betas = dict(np.load(shape_file))['betas']

    # Standard SMPL model
    gender = "male" if args.seqname.startswith("male") else "female"
    print(gender)

    body_model_smpl = SMPL(model_path="./data/SMPLX/smpl", gender=gender).cuda()

    # EasyMocap SMPL model
    body_model_em = SMPLlayerEM(
        "./data/SMPLX/smpl",
        model_type="smpl",
        gender=gender,
        device=device,
        regressor_path=os.path.join("data/smplh", "J_regressor_body25_smplh.txt"),
    )

    # Load AISt++ poses
    smpl_params = dict(np.load("load/animation/male-3-casual/poses.npz"))
    camera = dict(np.load("load/animation/male-3-casual/cameras.npz"))

    height = camera["height"]
    width = camera["width"]

    K = np.eye(3)
    K[0, 0] = K[1, 1] = 2000
    K[0, 2] = height // 2
    K[1, 2] = width // 2
    intrinsic = K.copy()

    extrinsic = camera["extrinsic"].copy()

    thetas = smpl_params["poses"][..., :72]
    transl = smpl_params["trans"] - smpl_params["trans"][0:1]
    transl += (0, 0.15, 5)

    # smpl_params = {
    #     "betas": betas.astype(np.float32).reshape(1, 10),
    #     "body_pose": thetas[..., 3:].astype(np.float32),
    #     "global_orient": thetas[..., :3].astype(np.float32),
    #     "transl": transl.astype(np.float32),
    # }

    global_orients = thetas[::args.skip, :3].astype(np.float32)
    transls = transl[::args.skip].astype(np.float32)
    body_poses = thetas[::args.skip, 3:].astype(np.float32)

    shape = betas.copy()
    global_orients_em = global_orients.copy()
    body_poses_em = np.concatenate(
        [np.zeros_like(body_poses[..., :3]), body_poses], axis=-1
    )
    transls_em = []

    for idx, (global_orient, body_pose, transl) in enumerate(
        zip(global_orients, body_poses, transls)
    ):
        print("Processing: {}".format(idx))

        # base_name = os.path.basename(img_file).split(".")[0]
        R = extrinsic[:3, :3].copy()
        T = extrinsic[:3, 3:].copy()

        poses_torch = torch.from_numpy(body_pose[None]).cuda()
        betas_torch = torch.from_numpy(betas).cuda()
        global_orient_torch = torch.from_numpy(global_orient[None]).cuda()
        transl_torch = torch.from_numpy(transl[None]).cuda()

        # SMPL mesh via standard SMPL
        smpl_outputs = body_model_smpl(
            betas=betas_torch,
            body_pose=poses_torch,
            global_orient=global_orient_torch,
            transl=transl_torch,
        )
        verts_smpl = smpl_outputs.vertices.detach().cpu().numpy()[0]

        poses_em_torch = torch.from_numpy(body_poses_em[idx][None]).cuda()

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
        transl_em = transl[None] + (verts_smpl - verts_em).mean(axis=0, keepdims=True)

        transls_em.append(transl_em)
        transl_em_torch = torch.from_numpy(transl_em).cuda()

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
                os.path.join(vertices_out_dir, "verts_{:04d}.npy".format(idx)),
                verts_smpl,
            )

            renderer = Renderer(
                height=height,
                width=width,
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

            images = [np.zeros((height, width, 3), dtype=np.uint8)]
            smpl_image = renderer.render(
                render_data,
                render_cameras,
                images,
                use_white=False,
                add_back=True,
            )[0]

            cv2.imwrite(
                os.path.join(vis_dir, f"{idx:04d}.png"),
                cv2.cvtColor(smpl_image, cv2.COLOR_RGB2BGR),
            )

            del renderer

    # Save easymocap SMPL parameters
    out_filename = os.path.join(out_dir, "params.npy")
    params = {
        "gender": gender,
        "beta": shape,
        "pose": np.concatenate(
            [
                global_orients_em,
                body_poses_em,
            ],
            axis=-1,
        ),
        "transl": np.concatenate(transls_em, axis=0),
    }
    np.save(
        out_filename,
        params,
    )
    # Save new camera parameters
    out_filename = os.path.join(out_dir, "cameras.npz")
    np.savez(
        out_filename,
        height=height,
        width=width,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
    )
