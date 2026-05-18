<div align="center">

# GR00T-VisualSim2Real


<table>
  <tr>
    <td align="center" valign="top" width="50%">
      <p><b>VIRAL</b><br><br><b>Visual Sim-to-Real at Scale for<br>Humanoid Loco-Manipulation</b></p>
      <a href="https://arxiv.org/abs/2511.15200"><img src="https://img.shields.io/badge/arXiv-2511.15200-b31b1b.svg" alt="VIRAL Paper"></a>
      <a href="https://viral-humanoid.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="VIRAL Project Page"></a>
      <a href="https://github.com/NVlabs/GR00T-VisualSim2Real/tree/main"><img src="https://img.shields.io/badge/Code-viral-lightgrey.svg" alt="VIRAL Code"></a>
      <br><br>
      <img src="./media/viral-teaser.gif" width="100%">
    </td>
    <td align="center" valign="top" width="50%">
      <p><b>DoorMan</b><br><br><b>Opening the Sim-to-Real Door for<br>Humanoid Pixel-to-Action Policy Transfer</b></p>
      <a href="https://arxiv.org/abs/2512.01061"><img src="https://img.shields.io/badge/arXiv-2512.01061-b31b1b.svg" alt="DoorMan Paper"></a>
      <a href="https://doorman-humanoid.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="DoorMan Project Page"></a>
      <a href="https://github.com/NVlabs/GR00T-VisualSim2Real/tree/doorman"><img src="https://img.shields.io/badge/Code-doorman-lightgrey.svg" alt="DoorMan Code"></a>
      <br><br>
      <img src="./media/doorman-teaser.gif" width="100%">
    </td>
  </tr>
</table>


</div>

<br/>

## Overview
This repository contains the application code for **VIRAL** (Visual Sim-to-Real for Humanoid Loco-Manipulation) and **DoorMan**. The system enables humanoid robots (e.g., Unitree G1) to perform complex tasks like opening heavy doors in the real world through a teacher-student simulation framework.

This repository contains the official code for:

- **VIRAL**: *Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation*
  &nbsp; [Paper](https://arxiv.org/abs/2511.15200) &nbsp;|&nbsp; [Project](https://viral-humanoid.github.io/) &nbsp;|&nbsp; [Code](https://github.com/NVlabs/GR00T-VisualSim2Real/tree/main)
- **DoorMan**: *Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer*
  &nbsp; [Paper](https://arxiv.org/abs/2512.01061) &nbsp;|&nbsp; [Project](https://doorman-humanoid.github.io/) &nbsp;|&nbsp; [Code](https://github.com/NVlabs/GR00T-VisualSim2Real/tree/doorman)

---


# DoorMan: Door-Opening Loco-Manipulation for Humanoid Robots

Reinforcement learning for humanoid robot door-opening on the Unitree G1 robot. This codebase supports:

- **Teacher training**: PPO with LSTM actor/critic using privileged state observations
- **Student training**: Vision-based policy distillation (DAgger) with ResNet18 + LSTM from a trained teacher
- **Evaluation**: Evaluate trained teacher or student checkpoints, with optional ONNX export

The task involves a 6-stage door-opening pipeline: walk to door, pregrasp, grasp handle, open handle, swing door open, and walk through.

Built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) (Isaac Sim 5.1), [TRL](https://github.com/huggingface/trl), and [Hydra](https://hydra.cc/) for configuration management.

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with driver >= 535
- Conda or Mamba

## Environment Setup

### 1. Create conda environment

```bash
conda create -n isaacsim5.1 python=3.11 -y
conda activate isaacsim5.1
```

### 2. Install Isaac Sim 5.1

```bash
# Install PyTorch with CUDA support
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Sim 5.1
pip install isaacsim==5.1.0.0 isaacsim-rl==5.1.0.0
```

### 3. Install Isaac Lab

Clone or download [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and install:

```bash
# Pre-install build dependencies
pip install setuptools poetry-core flatdict

# Install Isaac Lab core and extensions
pip install --no-build-isolation -e <path-to-IsaacLab>/source/isaaclab
pip install --no-build-isolation -e <path-to-IsaacLab>/source/isaaclab_assets \
    -e <path-to-IsaacLab>/source/isaaclab_tasks \
    -e "<path-to-IsaacLab>/source/isaaclab_rl[all]"

# Fix numpy version to match Isaac Sim requirements
pip install numpy==1.26.0
```

Verify `isaaclab` is importable:

```bash
python -c "import isaaclab; print(isaaclab.__file__)"
```

### 4. Install this package

```bash
cd <path-to-this-repo>
pip install -e .

# Fix numpy version again (pip may upgrade it)
pip install numpy==1.26.0
```

### 5. Download LAFAN-G1 motion dataset

The door task uses the LAFAN-G1 dataset for robot reset initialization. Download it from Hugging Face:

```bash
# Install git-lfs if not already installed
sudo apt install git-lfs
git lfs install

# Clone the dataset into the same parent directory as this repo
# e.g., if this repo is at ~/projects/VIRAL, clone to ~/projects/LAFAN-G1
cd <parent-directory-of-this-repo>
git clone https://huggingface.co/datasets/ember-lab-berkeley/LAFAN-G1
```

The LAFAN-G1 directory must be at the same folder level as this repo (i.e., sibling directories). The expected layout is:

```
projects/
├── VIRAL/          # This repo
└── LAFAN-G1/       # LAFAN-G1 dataset
```

### 6. Download HOMIE locomotion models

The door task uses HOMIE locomotion models for lower-body control. Place the following model files in the repo root `./models/` directory:
- `model_walk.pt` - Walking locomotion policy
- `model_stand.pt` - Standing locomotion policy

### 7. Verify installation

```bash
python -c "from gr00t.rl.envs.door.door_open_homie import DoorPregrasp; print('OK')"
```

## Project Structure

```
gr00t/rl/
├── train_agent_trl.py          # Training entry point (teacher & student)
├── eval_agent_trl.py           # Evaluation entry point
├── config/                     # Hydra YAML configs
│   ├── base.yaml               # Base training config
│   ├── base_eval.yaml          # Base evaluation config
│   ├── exp/wbmanip/            # Door experiment configs
│   ├── algo/                   # Algorithm configs (PPO, DAgger)
│   ├── env/                    # Environment configs
│   ├── robot/g1/               # Robot configs (G1 43-DOF)
│   ├── rewards/wbmanip/        # Reward function configs
│   ├── obs/wbmanip/            # Observation configs
│   └── domain_rand/            # Domain randomization configs
├── envs/                       # Environment implementations
│   ├── base_task/              # Base task classes (staged, delta, homie, finger primitive)
│   ├── door/                   # Door-opening task (DoorPregrasp)
│   └── legged_base_task/       # Legged robot base
├── trl/                        # TRL-based trainers and modules
│   ├── trainer/                # PPO and distillation trainers
│   ├── modules/                # Actor-critic network modules (MLP, LSTM, Vision)
│   ├── callbacks/              # Training callbacks (save, eval, wandb)
│   └── utils/                  # Training utilities
├── agents/modules/             # Neural network building blocks (MLP, CNN, ResNet)
├── simulator/isaacsim/         # Isaac Sim interface
├── data/                       # Task data
│   ├── robots/g1/              # G1 robot USD assets
│   ├── motions/g1_wsg/         # Door demonstration motion data
│   ├── objects/grab/           # Door handle assets
│   └── tasks/door/             # Door scenario configuration
└── utils/                      # General utilities
```

## Door Asset Generation

During training, door assets are procedurally generated on the fly. You can also generate them **offline** — see [gr00t/rl/scripts/README.md](gr00t/rl/scripts/README.md) for full documentation.

<p align="center">
  <img src="./media/door_assets.gif" width="90%">
</p>

```bash
# Quick start: generate 100 doors matching training config
python gr00t/rl/scripts/generate_door_assets.py \
    --num_doors 100 --output_dir data/door_assets \
    --build_latch --add_floors --randomize_material --seed 42

# Generate 1000 doors with diverse configurations
bash gr00t/rl/scripts/generate_1000_doors.sh
```

## Usage

### Teacher Training (PPO + LSTM)

Train a teacher policy using privileged state observations with LSTM memory. The experiment config is at [`gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml`](gr00t/rl/config/exp/wbmanip/door_open_homie_lstm.yaml).

```bash
python gr00t/rl/train_agent_trl.py \
    +exp=wbmanip/door_open_homie_lstm \
    ++num_envs=1024 \
    ++algo.config.entropy_coef=0.001 \
    ++algo.config.num_steps_per_env=32 \
    ++env.config.delta_action_scale=0.3
```

Key arguments:
- `num_envs`: Number of parallel environments (1024 recommended; reduce if GPU memory limited)
- `algo.config.entropy_coef`: Entropy coefficient for exploration
- `algo.config.num_steps_per_env`: Rollout length per environment per update
- `env.config.delta_action_scale`: Scale for delta action space

### Teacher Evaluation

Evaluate a trained teacher checkpoint:

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt>
```

For example:
```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=logs_rl/g1_open_door_homie/wbmanip/door_open_homie_lstm-20250101_120000/model_step_020000.pt
```

### Student Training (DAgger + ResNet18 + LSTM)

Train a vision-based student policy by distilling from a trained teacher. The experiment config is at [`gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml`](gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml).

```bash
python gr00t/rl/train_agent_trl.py \
    +exp=wbmanip/door_open_homie_dagger-lstm \
    ++num_envs=64 \
    ++algo.config.num_steps_per_env=32 \
    ++algo.config.actor.backbone.vision_module.module_config_dict.layer_config.trainable=True \
    ++algo.config.obj_pred_loss_coef=1.0 \
    ++algo.config.actor.running_mean_std=True \
    ++algo.config.num_learning_epochs=1 \
    ++algo.config.num_mini_batches=64 \
    ++algo.config.teacher_rollout_ratio=0.3
```

**Important**: Before running student training, update the `teacher_actor_path` in the experiment config to point to your trained teacher checkpoint:

```yaml
# In gr00t/rl/config/exp/wbmanip/door_open_homie_dagger-lstm.yaml
teacher_actor_path: logs_rl/<your_teacher_experiment>/model_step_XXXXXX.pt
```

Key student training arguments:
- `num_envs`: Number of parallel environments (64 recommended for vision training)
- `algo.config.teacher_rollout_ratio`: Fraction of rollouts using the teacher policy (curriculum)
- `algo.config.obj_pred_loss_coef`: Weight for object position prediction auxiliary loss
- `algo.config.actor.backbone.vision_module.module_config_dict.layer_config.trainable`: Whether to fine-tune ResNet18 backbone

### Student Evaluation

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=logs_rl/<student_experiment_dir>/model_step_XXXXXX.pt
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration. Configs are composed from YAML files in `gr00t/rl/config/`. Override any config value from the command line with `++`:

```bash
# Override number of environments
... ++num_envs=16

# Override reward weights
... ++rewards.reward_scales.walk_to_door=10.0

# Override training hyperparameters
... ++algo.config.actor_learning_rate=1e-4
```

## Experiment Tracking

Training logs to [Weights & Biases](https://wandb.ai/) by default. Set your wandb credentials:

```bash
wandb login
```

Checkpoints are saved to `logs_rl/<project_name>/<experiment_name>/` with periodic model saves controlled by the `save_interval` parameter (default: 500 steps).

## ONNX Export

During evaluation with `num_envs=1`, the policy can be exported as ONNX for deployment:

```bash
python gr00t/rl/eval_agent_trl.py \
    +checkpoint=<path_to_checkpoint.pt> \
    num_envs=1
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Copyright & Attribution

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This project includes third-party open-source software. Please refer to individual source files for specific licenses and copyright headers.
