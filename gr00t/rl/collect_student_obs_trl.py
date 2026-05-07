# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect per-frame student observations and raw camera images during evaluation.

Example:
    python gr00t/rl/collect_student_obs_trl.py \
        +checkpoint=logs_rl/<student_experiment_dir>/model_step_XXXXXX.pt \
        num_envs=8 \
        +collect_student_obs.output_dir=logs_eval/student_obs/<run_name> \
        +collect_student_obs.num_episodes=32
"""

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from gr00t.rl.eval_agent_trl import process_output_dim_in_config
from gr00t.rl.utils.config_utils import register_rl_resolvers
from gr00t.rl.utils.student_obs_collection import collect_student_obs

register_rl_resolvers()


def _load_eval_config(override_config: OmegaConf):
    if override_config.checkpoint is not None:
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"

        if config_path.exists():
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)
            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
            config = OmegaConf.merge(train_config, override_config)
        else:
            logger.error(f"Could not find config path near checkpoint: {checkpoint}")
            config = override_config
        config.experiment_dir = checkpoint.parent
        return config

    if override_config.eval_overrides is not None:
        config = override_config.copy()
        eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
        for arg in sys.argv[1:]:
            if not arg.startswith("+"):
                key = arg.split("=")[0]
                if key in eval_overrides:
                    del eval_overrides[key]
        config.eval_overrides = OmegaConf.create(eval_overrides)
        return OmegaConf.merge(config, eval_overrides)
    return override_config


def _setup_isaac_app(config):
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type != "IsaacSim":
        return None

    try:
        with open("./rl/simulator/isaacsim/.isaacsim_version", "r", encoding="utf-8") as f:
            default_isaacsim_version = f.read().strip()
    except FileNotFoundError:
        default_isaacsim_version = "4.5"

    if default_isaacsim_version == "4.5":
        from isaaclab.app import AppLauncher
    elif default_isaacsim_version == "4.2":
        logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
        from omni.isaac.lab.app import AppLauncher
    else:
        raise ValueError(f"Unsupported IsaacSim version: {default_isaacsim_version}")

    import argparse

    import isaaclab

    parser = argparse.ArgumentParser(description="Collect student observations.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    args_cli.num_envs = config.num_envs
    args_cli.seed = config.seed
    args_cli.env_spacing = config.env.config.env_spacing
    args_cli.output_dir = config.output_dir
    args_cli.enable_cameras = True
    args_cli.headless = config.headless

    dest_path = Path(isaaclab.__file__).resolve().parent.parent.parent.parent / "apps"
    current_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    if args_cli.headless:
        source_file = current_file_dir_path / "apps/phc.isaaclab.python.headless.rendering.kit"
        shutil.copy(source_file, dest_path)
        args_cli.experience = dest_path / "phc.isaaclab.python.headless.rendering.kit"

    return AppLauncher(args_cli).app


def _get_collect_kwargs(config, checkpoint: Path):
    collect_cfg = OmegaConf.select(config, "collect_student_obs")
    collect = OmegaConf.to_container(collect_cfg, resolve=True) if collect_cfg else {}
    if collect is None:
        collect = {}

    if not collect.get("output_dir"):
        ckpt_step = checkpoint.stem
        collect["output_dir"] = str(
            Path("logs_eval")
            / "student_obs"
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{checkpoint.parent.name}-{ckpt_step}"
        )

    return collect


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "collect_student_obs.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    from gr00t.rl.utils.logging import HydraLoggerBridge

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())
    os.chdir(hydra.utils.get_original_cwd())

    config = _load_eval_config(override_config)
    checkpoint = Path(config.checkpoint)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path, "r"))
        config.wandb.wandb_id = meta["wandb_run"]
        print(f"resume wandb from run: {config.wandb.wandb_id}")

    config.simulator.config.cameras.enable_cameras = True
    simulation_app = _setup_isaac_app(config)

    from accelerate import Accelerator
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    from gr00t.rl.agents.modules.ppo_modules import (
        PPOCritic,
        PPOStateActor,
        PPOStateActorFixSigma,
    )
    from gr00t.rl.trl.utils.common import custom_instantiate
    from gr00t.rl.utils.common import seeding
    from gr00t.rl.utils.helpers import pre_process_config

    os.chdir(hydra.utils.get_original_cwd())
    pre_process_config(config)
    config.env.config.ckpt_dir = str(checkpoint.parent)

    config.algo.trl.output_dir = str(Path(config.experiment_dir))
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    _script_args, training_args, _model_args = parser.parse_dict(config.algo.trl)
    if "eval_output_dir" in config:
        training_args.eval_output_dir = config.eval_output_dir

    accelerator = Accelerator()
    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    env = instantiate(config=config.env, device=device)
    process_output_dim_in_config(config)

    ref_model = None
    value_model = None
    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy = instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            _recursive_=False,
        ).to(device)
        if getattr(config.algo.config, "use_dagger", False):
            ref_model = instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
                input_key="teacher_obs",
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False) and hasattr(
            config.algo.config, "critic"
        ):
            value_model = instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _recursive_=False,
            ).to(device)
    else:
        if getattr(config.algo.config, "use_dagger", False):
            module_dim_dict = getattr(config.algo.config, "module_dim", {})
            policy = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
            ref_model = PPOStateActorFixSigma(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.teacher_actor,
                num_actions=env.config.robot.actions_dim,
                module_dim_dict=module_dim_dict,
            ).to(device)
        else:
            policy = PPOStateActor(
                obs_dim_dict=env.config.robot.algo_obs_dim_dict,
                module_config_dict=config.algo.config.module_dict.actor,
                num_actions=env.config.robot.actions_dim,
                input_key="actor_obs",
                init_noise_std=config.algo.config.init_noise_std,
            ).to(device)
            value_model = PPOCritic(
                env.config.robot.algo_obs_dim_dict,
                config.algo.config.module_dict.critic,
            ).to(device)

    accelerator.wait_for_everyone()

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))

    trainer = custom_instantiate(
        config.trainer,
        args=training_args,
        config=config.algo.config,
        env=env,
        model=policy,
        ref_model=ref_model,
        use_ref_model=getattr(config.algo.config, "use_dagger", False),
        value_model=value_model,
        train_dataset=None,
        eval_dataset=None,
        callbacks=callbacks,
        checkpoint=config.checkpoint,
        local_seed=config.seed,
        accelerator=accelerator,
    )

    collect_kwargs = _get_collect_kwargs(config, checkpoint)
    output_dir = collect_kwargs.pop("output_dir")
    logger.info(f"Collecting student observations to {output_dir}")
    collect_student_obs(
        trainer,
        output_dir=output_dir,
        checkpoint=str(checkpoint),
        **collect_kwargs,
    )
    logger.info("Finished student observation collection")

    if simulation_app is not None:
        simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
