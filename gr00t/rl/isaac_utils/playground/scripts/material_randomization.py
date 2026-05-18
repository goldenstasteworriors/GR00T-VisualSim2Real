# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import numpy as np
import omni.kit.commands
from isaaclab.sim.utils import bind_visual_material
from pxr import Sdf, Usd
from typing_extensions import override

from gr00t.rl.isaac_utils.playground.utils.usd_utils import (
    get_custom_data_from_prim,
    write_custom_data_to_prim,
)

MATERIAL_RANDOMIZATION_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "material_randomization.py"
)


def _get_behavior_script_base():
    """Lazy import: omni.kit.scripting requires its extension to be enabled first."""
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.kit.scripting", True)

    from omni.kit.scripting import BehaviorScript

    return BehaviorScript


# Build the class lazily so that the module can be imported before SimulationApp init.
# The actual class is created on first access via get_material_randomization_class().
_MaterialRandomization = None


def get_material_randomization_class():
    """Return the MaterialRandomization class, creating it on first call."""
    global _MaterialRandomization
    if _MaterialRandomization is not None:
        return _MaterialRandomization

    BehaviorScript = _get_behavior_script_base()

    class MaterialRandomization(BehaviorScript):
        @override
        def on_init(self):
            custom_data = get_custom_data_from_prim(self.stage, self.prim.GetPrim().GetPath())
            self.randomization_interval = custom_data["randomizationInterval"]
            self.randomization_material_list = custom_data["randomizationMaterialList"]
            self.last_randomization_time = 0.0
            self.randomization_interval_perturbation = np.random.uniform(
                self.randomization_interval * -0.1, self.randomization_interval * 0.1
            )

        @override
        def on_update(self, current_time: float, delta_time: float):
            if (
                current_time - self.last_randomization_time
                < self.randomization_interval + self.randomization_interval_perturbation
            ):
                return

            self.last_randomization_time = current_time
            self.randomization_interval_perturbation = np.random.uniform(
                self.randomization_interval * -0.1, self.randomization_interval * 0.1
            )

            new_material_prim_path = np.random.choice(self.randomization_material_list)
            bind_visual_material(self.prim.GetPrim().GetPath(), new_material_prim_path, self.stage)

        @override
        def on_play(self):
            self.last_randomization_time = 0.0

        @staticmethod
        def add_to_prim(
            stage: Usd.Stage,
            prim_path: str,
            randomization_interval: float,
            randomization_material_list: list[str],
        ):
            write_custom_data_to_prim(
                stage,
                prim_path,
                {
                    "randomizationInterval": randomization_interval,
                    "randomizationMaterialList": randomization_material_list,
                },
            )
            omni.kit.commands.execute("ApplyScriptingAPICommand", paths=[Sdf.Path(prim_path)])
            stage.GetPrimAtPath(prim_path).CreateAttribute(
                "omni:scripting:scripts", Sdf.ValueTypeNames.AssetArray
            ).Set([MATERIAL_RANDOMIZATION_SCRIPT_PATH])

    _MaterialRandomization = MaterialRandomization
    return _MaterialRandomization
