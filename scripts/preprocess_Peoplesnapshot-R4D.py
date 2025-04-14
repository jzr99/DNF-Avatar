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

parser = argparse.ArgumentParser(description="Preprocessing for Peoplesnapshot.")
parser.add_argument(
    "--data-dir", type=str, help="Directory that contains Peoplesnapshot data in InstantAvatar format."
)
# parser.add_argument(
#     "--split",
#     type=str,
#     help="Split to process.",
#     choices=["train", "val", "test"],
# )
parser.add_argument(
    "--out-dir", type=str, help="Directory where preprocessed data is saved."
)
parser.add_argument(
    "--seqname", type=str, default="male-3-casual", help="Sequence to process."
)
parser.add_argument("--visualize", action="store_true", help="Visualize SMPL mesh.")

if __name__ == "__main__":
    args = parser.parse_args()
    seq_name = args.seqname
    data_dir = os.path.join(args.data_dir, seq_name)
    out_dir = os.path.join(args.out_dir, seq_name)

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda")

    if args.visualize:
        vis_dir = os.path.join(out_dir, "smpl_vis", "images")
        os.makedirs(vis_dir, exist_ok=True)

    vertices_out_dir = os.path.join(out_dir, "vertices")
    os.makedirs(vertices_out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(data_dir, "images", "image_*.png")))
    if args.seqname == "male-3-casual":
        img_files = img_files[:456:4] + img_files[456:457:4] + img_files[456:676:4]
    elif args.seqname == "male-4-casual":
        img_files = img_files[:660:6] + img_files[660:661:6] + img_files[660:873:6]
    elif args.seqname == "female-3-casual":
        img_files = img_files[:446:4] + img_files[446:447:4] + img_files[446:648:4]
    elif args.seqname == "female-4-casual":
        img_files = img_files[:336:4] + img_files[335:336:4] + img_files[335:524:4]

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

    poses = {}
    for split in ["train", "val", "test"]:
        poses_ = dict(np.load(os.path.join(data_dir, "poses", f"anim_nerf_{split}.npz")))
        for k, v in poses_.items():
            if k in ["global_orient", "transl", "body_pose"]:
                if poses.get(k) is None:
                    poses[k] = v
                else:
                    poses[k] = np.concatenate([poses.get(k), v], axis=0)
            else:
                poses[k] = v

    camera = dict(np.load(os.path.join(data_dir, "cameras.npz")))

    global_orients = poses["global_orient"]
    transls = poses["transl"]
    body_poses = poses["body_pose"]
    betas = poses["betas"]

    shape = betas.copy()
    global_orients_em = global_orients.copy()
    body_poses_em = np.concatenate(
        [np.zeros_like(body_poses[..., :3]), body_poses], axis=-1
    )
    transls_em = []

    for idx, img_file in enumerate(img_files):
        if idx >= len(poses["global_orient"]):
            break

        print("Processing: {}".format(img_file))

        base_name = os.path.basename(img_file).split(".")[0]

        # Intrinsics
        intrinsic = camera["intrinsic"]
        # Extrinsics
        extrinsic = camera["extrinsic"]
        R = extrinsic[:3, :3].copy()
        T = extrinsic[:3, 3:].copy()

        poses_torch = torch.from_numpy(body_poses[idx][None]).cuda()
        betas_torch = torch.from_numpy(betas).cuda()
        global_orient_torch = torch.from_numpy(global_orients[idx][None]).cuda()
        transl_torch = torch.from_numpy(transls[idx][None]).cuda()

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
        transl_em = transls[idx][None] + (verts_smpl - verts_em).mean(
            axis=0, keepdims=True
        )

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
    # Copy image and mask folders
    shutil.copytree(
        os.path.join(data_dir, "images"),
        os.path.join(out_dir, "images"),
    )
    shutil.copytree(
        os.path.join(data_dir, "masks"),
        os.path.join(out_dir, "masks"),
    )
    # Copy camera parameters
    shutil.copy(
        os.path.join(data_dir, "cameras.npz"),
        os.path.join(out_dir, "cameras.npz"),
    )
