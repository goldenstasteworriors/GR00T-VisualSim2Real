import json
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def _detach_cpu(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detach_cpu(v) for v in value)
    return value


def _slice_env(value: Any, env_id: int, num_envs: int):
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == num_envs:
            return value[env_id].detach().cpu()
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _slice_env(v, env_id, num_envs) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_slice_env(v, env_id, num_envs) for v in value)
    return value


def _slice_feature(value: torch.Tensor, env_id: int, num_envs: int):
    value = value.detach()
    if value.ndim == 0:
        return value.cpu()
    if value.shape[0] == num_envs:
        return value[env_id].cpu()
    if value.shape[0] % num_envs == 0:
        return value.reshape(num_envs, -1, *value.shape[1:])[env_id].cpu()
    return value.cpu()


def _to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr)
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _save_rgb_png(image: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_rgb_uint8(image)
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except Exception:
        np.save(path.with_suffix(".npy"), arr)


def _save_depth(depth: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = depth.detach().cpu().numpy()
    arr = np.nan_to_num(arr, posinf=0.0, neginf=0.0)
    np.save(path, arr)


def _snapshot_camera_outputs(env, num_envs: int):
    simulator = getattr(env, "simulator", None)
    scene = getattr(simulator, "scene", None)
    sensors = getattr(scene, "sensors", {}) if scene is not None else {}

    camera_outputs = {}
    for sensor_name, sensor in sensors.items():
        data = getattr(sensor, "data", None)
        outputs = getattr(data, "output", None)
        if not isinstance(outputs, dict):
            continue
        for output_name, tensor in outputs.items():
            if not torch.is_tensor(tensor) or tensor.ndim == 0 or tensor.shape[0] != num_envs:
                continue
            safe_name = f"{sensor_name}_{output_name}"
            camera_outputs[safe_name] = {
                "type": output_name,
                "tensor": tensor.detach().cpu(),
            }
    return camera_outputs


class StudentActorFeatureRecorder:
    def __init__(self, policy):
        self.policy = policy
        self.features: Dict[str, torch.Tensor] = {}
        self.handles = []

    def __enter__(self):
        self._register_hook("vision_features", getattr(self.policy, "vision_module", None))
        self._register_hook(
            "state_history_features", getattr(self.policy, "state_history_module", None)
        )
        self._register_pre_hook(
            "student_actor_mlp_input", getattr(self.policy, "mlp_module", None)
        )
        self._register_pre_hook("object_prediction_input", getattr(self.policy, "obj_pred_mlp", None))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self):
        self.features.clear()

    def _register_hook(self, name: str, module):
        if module is None:
            return

        def hook(_module, _inputs, output):
            if torch.is_tensor(output):
                self.features[name] = output.detach()

        self.handles.append(module.register_forward_hook(hook))

    def _register_pre_hook(self, name: str, module):
        if module is None:
            return

        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                self.features[name] = inputs[0].detach()

        self.handles.append(module.register_forward_pre_hook(hook))


class StudentObsEpisodeWriter:
    def __init__(
        self,
        output_dir: str,
        num_envs: int,
        checkpoint: Optional[str] = None,
        save_camera_tensors: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.num_envs = num_envs
        self.save_camera_tensors = save_camera_tensors
        self.global_episode_id = 0
        self.env_episode_ids = [0 for _ in range(num_envs)]
        self.env_frame_ids = [0 for _ in range(num_envs)]
        self.env_episode_dirs = [self._new_episode_dir(env_id) for env_id in range(num_envs)]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_dataset_metadata(checkpoint)

    def _write_dataset_metadata(self, checkpoint: Optional[str]):
        metadata = {
            "checkpoint": checkpoint,
            "layout": (
                "episode_%06d_env_%03d/frame_%06d/{metadata.json,"
                "student_actor_input.pt,action_state.pt,cameras/}"
            ),
        }
        with open(self.output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _new_episode_dir(self, env_id: int) -> Path:
        episode_id = self.global_episode_id
        self.global_episode_id += 1
        self.env_episode_ids[env_id] = episode_id
        self.env_frame_ids[env_id] = 0
        episode_dir = self.output_dir / f"episode_{episode_id:06d}_env_{env_id:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        return episode_dir

    def write_frame(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_state: Dict[str, Any],
        features: Dict[str, torch.Tensor],
        camera_outputs: Dict[str, Dict[str, torch.Tensor]],
        env_id: int,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        frame_id = self.env_frame_ids[env_id]
        frame_dir = self.env_episode_dirs[env_id] / f"frame_{frame_id:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        student_actor_input = {
            "obs": _slice_env(obs_dict, env_id, self.num_envs),
            "features": {
                k: _slice_feature(v, env_id, self.num_envs) for k, v in features.items()
            },
        }
        torch.save(student_actor_input, frame_dir / "student_actor_input.pt")
        torch.save(_slice_env(action_state, env_id, self.num_envs), frame_dir / "action_state.pt")
        self._write_camera_outputs(camera_outputs, env_id, frame_dir / "cameras")

        metadata = {
            "env_id": env_id,
            "episode_id": self.env_episode_ids[env_id],
            "frame_id": frame_id,
            "reward_after_action": float(reward[env_id].detach().cpu()) if reward is not None else None,
            "done_after_action": bool(done[env_id].detach().cpu()) if done is not None else None,
        }
        if info is not None:
            metadata["info_keys"] = sorted(info.keys())
        with open(frame_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.env_frame_ids[env_id] += 1

    def finish_episodes(self, env_ids):
        for env_id in env_ids:
            self.env_episode_dirs[int(env_id)] = self._new_episode_dir(int(env_id))

    def _write_camera_outputs(self, camera_outputs, env_id: int, camera_dir: Path):
        raw_tensors = {}
        for safe_name, payload in camera_outputs.items():
            tensor = payload["tensor"]
            if not torch.is_tensor(tensor) or tensor.ndim == 0 or tensor.shape[0] <= env_id:
                continue
            output_name = payload["type"]
            env_tensor = tensor[env_id]
            if output_name == "rgb":
                _save_rgb_png(env_tensor, camera_dir / f"{safe_name}.png")
            elif output_name == "depth":
                _save_depth(env_tensor, camera_dir / f"{safe_name}.npy")
            if self.save_camera_tensors:
                raw_tensors[safe_name] = env_tensor.detach().cpu()

        if raw_tensors:
            torch.save(raw_tensors, camera_dir / "raw_camera_tensors.pt")


def _get_generation_context(trainer):
    if hasattr(trainer, "accelerator") and trainer.accelerator:
        try:
            from trl.trainer.ppo_trainer import unwrap_model_for_generation

            return unwrap_model_for_generation(
                trainer.model,
                trainer.accelerator,
                gather_deepspeed3_params=getattr(
                    trainer.args, "ds3_gather_for_generation", False
                ),
            )
        except Exception as exc:
            logger.warning(f"Falling back to direct model context: {exc}")
    return nullcontext(trainer.model)


def collect_student_obs(trainer, output_dir: str, checkpoint: Optional[str] = None, **kwargs):
    env = trainer.env
    device = (
        trainer.accelerator.device
        if getattr(trainer, "accelerator", None)
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    max_episodes = int(kwargs.get("num_episodes") or trainer.config.eval.get("num_eval_episodes", env.num_envs))
    eval_num_envs_episodes = bool(kwargs.get("eval_num_envs_episodes", False))
    save_camera_tensors = bool(kwargs.get("save_camera_tensors", False))

    trainer._eval_mode()
    trainer.policy_model.eval_mode()
    trainer.policy_model.init_rollout()
    obs_dict = env.reset_all()
    for obs_key in obs_dict.keys():
        obs_dict[obs_key] = obs_dict[obs_key].to(device)

    env.init_eval_metrics_tracking(device)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.int32, device=device)
    completed_once = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    writer = StudentObsEpisodeWriter(
        output_dir=output_dir,
        num_envs=env.num_envs,
        checkpoint=checkpoint,
        save_camera_tensors=save_camera_tensors,
    )

    def should_stop(collected_episodes: int):
        if eval_num_envs_episodes:
            return torch.all(completed_once).item()
        return collected_episodes >= max_episodes

    collected_episodes = 0
    pbar_total = env.num_envs if eval_num_envs_episodes else max_episodes
    pbar = tqdm(total=pbar_total)
    dones = torch.zeros(env.num_envs, device=device)
    info_by_env: Dict[int, Dict[str, Any]] = defaultdict(dict)

    with torch.no_grad(), _get_generation_context(trainer) as model:
        with StudentActorFeatureRecorder(model.policy) as recorder:
            while not should_stop(collected_episodes):
                recorder.clear()
                action_state = trainer.policy_step(model.policy, obs_dict, cur_dones=dones)
                action_state["actions"] = action_state["action_mean"]
                frame_obs_dict = _detach_cpu(obs_dict)
                camera_outputs = _snapshot_camera_outputs(env, env.num_envs)

                env.render_results()
                next_obs_dict, rewards, dones, infos = env.step(action_state)

                rewards = rewards.to(device)
                dones = dones.to(device)
                cur_reward_sum += rewards
                cur_episode_length += 1
                env.update_eval_metrics_per_step(infos)

                done_env_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                for env_id in range(env.num_envs):
                    writer.write_frame(
                        obs_dict=frame_obs_dict,
                        action_state=action_state,
                        features=recorder.features,
                        camera_outputs=camera_outputs,
                        env_id=env_id,
                        reward=rewards,
                        done=dones,
                        info=info_by_env.get(env_id),
                    )

                if len(done_env_ids) > 0:
                    if eval_num_envs_episodes:
                        valid_new_ids = done_env_ids[~completed_once[done_env_ids]]
                    else:
                        valid_new_ids = done_env_ids

                    if len(valid_new_ids) > 0:
                        env.process_eval_episode_completions(
                            valid_new_ids.unsqueeze(-1), cur_reward_sum, cur_episode_length
                        )
                        if eval_num_envs_episodes:
                            completed_once[valid_new_ids] = True
                        collected_episodes += int(len(valid_new_ids))
                        pbar.update(int(len(valid_new_ids)))

                    cur_reward_sum[done_env_ids] = 0
                    cur_episode_length[done_env_ids] = 0
                    env.reset_eval_episode_tracking(done_env_ids.unsqueeze(-1))
                    if not should_stop(collected_episodes):
                        writer.finish_episodes(done_env_ids.detach().cpu().tolist())

                obs_dict = next_obs_dict
                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(device)
                torch.cuda.empty_cache()

    env.end_render_results()
    pbar.close()

    eval_dict = env.get_eval_metrics_summary()
    with open(Path(output_dir) / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_dict, f, indent=2)
    logger.info(f"Saved student obs collection to {output_dir}")
    return eval_dict
