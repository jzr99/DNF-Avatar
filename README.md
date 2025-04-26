## ___***DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting***___
<!-- <div align="center"> -->

## [Paper](https://arxiv.org/pdf/2504.10486) | [Video Youtube](https://youtu.be/C4Q5U8w9X5U) | [Project Page](https://jzr99.github.io/DNF-Avatar)


<p align="center">
<img src='assets/dnf-avatar-logo.png' style="height:80px"></img>
</p>

<p align="center">
<img src="assets/m1_fps10_compress.gif" width="800" /> 
</p>

## ⚙️ Setup
### Code and SMPL Setup
* Clone the repository
```bash
git clone --recursive https://github.com/jzr99/DNF-Avatar.git
```
* Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Rename these `.pkl` files and copy them to subdirectories under `./data/SMPLX/smpl/`. Eventually, the `./data` folder should have the following structure:
```
data
 └-- SMPLX
    └-- smpl
       ├-- SMPL_FEMALE.pkl
       ├-- SMPL_MALE.pkl
       └-- SMPL_NEUTRAL.pkl
```

* Download some extracted SMPL parameter used in our code:

```bash
mkdir body_models
gdown https://drive.google.com/drive/folders/1HVJW51gRneFVsEBOj3j-OQy-rtrpe9eV -O ./body_models/ --folder
gdown https://drive.google.com/drive/folders/1dzHdqOQ4ZG71mmUGi1f1Q89dCvFWinF3 -O ./body_models/ --folder
gdown https://drive.google.com/drive/folders/1RTu6AalRezycGAbQimKMgMVtmFxneP2d -O ./ --folder
```

### Environment Setup
- Create a Python virtual environment via either `venv` or `conda`
- Install PyTorch>=1.13 [here](https://pytorch.org/get-started/locally/) based on the package management tool you are using and your cuda version (older PyTorch versions may work but have not been tested)
- Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- Install other packages: `pip install -r requirements.txt`
- Install gs-ir packages: `cd gs-ir && python setup.py develop && cd ..`
- Set `PYTHONPATH` to the current working directory: `export PYTHONPATH=${PWD}`


## Dataset Preparation
Please follow the steps in [DATASET.md](DATASET.md).

## Teacher Training
Use `launch_ps.py` for PeopleSnapshot Dataset, use `launch_rana.py` for RANA Dataset.

To train on the `male-3-casual` sequence of PeopleSnapshot, use the following command:
```
python launch_ps.py training_mode=teacher \
   implicit_branch.mode=train \
   implicit_branch/dataset=peoplesnapshot/male-3-casual \
   implicit_branch.trainer.max_steps=25000 \
   implicit_branch.tag=male-3-casual
```
Checkpoints and visualizations will be saved under the directory `exp/intrinsic-avatar-male-3-casual/male-3-casual`

## Generate Distillation Avatar
To generate pseudo ground truth image used for distillation, use the following command:
```
python launch_ps.py training_mode=teacher \
   implicit_branch.mode=test \
   implicit_branch.resume="exp/intrinsic-avatar-male-3-casual/male-3-casual/ckpt/epoch\=219-step\=25000.ckpt" \
   implicit_branch/dataset=animation/male-3-casual-tpose \
   implicit_branch.tag=male-3-casual-tpose
```
Then, the generated distillation avatar will be saved under the directory `exp/intrinsic-avatar-male-3-casual/male-3-casual-tpose`

## Student Training

To start distill knowledge from teacher model to student model, use the following command:
```
python launch_ps.py training_mode=distill \
   implicit_branch.mode=train \
   implicit_branch.resume="exp/intrinsic-avatar-male-3-casual/male-3-casual/ckpt/epoch\=219-step\=25000.ckpt" \
   implicit_branch/dataset=peoplesnapshot/male-3-casual \
   explicit_branch/dataset=ps_IA_male_3 \
   explicit_branch/distill_dataset=ps_IA_distill_male_3
```
The occlusion probe will be store in `occ_dir`, and the experiments results and checkpoints will be stored in `exp_dir`. Those two argruments and some other parameters are defined in `configs/explicit_branch/dataset/ps_IA_male_3.yaml`. Note that `distill_pose_data_dir` and `alpha_data_dir` should point to the output folder of the generated distillation avatar, and those argruments are defined in `configs/explicit_branch/distill_dataset/ps_IA_distill_male_3.yaml`.

## Testing

To start animating and relighting the student avatar, use the following command:

```
python test_ps.py \
   explicit_branch.mode=test \
   explicit_branch/dataset=ps_IA_animation_male_3 \
   explicit_branch.load_ckpt="exp/ps_male3_2dgs/ckpt30000.pth" \
   +explicit_branch.hdri="Environment_Maps/cambridge_2k.hdr"
```

To test the rendering speed, please use the `test_speed.py` file.

<p align="center">
<img src="assets/intrinsics_final.gif" width="800" /> 
</p>

## Acknowledgement
We have used codes from other great research work, including [IntrinsicAvatar](https://github.com/taconite/IntrinsicAvatar), [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release), [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [GS-IR](https://github.com/lzhnb/GS-IR), and [nvdiffrec](https://github.com/NVlabs/nvdiffrec). We sincerely thank the authors for their awesome work!

## BibTeX
If you find DNF-Avatar useful for your research and applications, please cite us using this BibTex:
```bibtex
@misc{DNF-Avatar,
      title={DNF-Avatar: Distilling Neural Fields for Real-time Animatable Avatar Relighting}, 
      author={Jiang, Zeren and Wang, Shaofei and Tang, Siyu},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
```


