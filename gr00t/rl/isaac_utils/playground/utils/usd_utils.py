# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import asyncio
import numpy as np
from pxr import Usd, Sdf, Gf, UsdPhysics, UsdGeom, UsdLux
import omni.kit.commands
import omni.kit.app
manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("omni.kit.material.library", True)
import omni.kit.material.library


def preload_materials(stage: Usd.Stage, material_list: list[str], material_root_prim_path: str, random_transform_each_num: int = 1, random_color_material_list: list[str] | None = None, random_color_each_num: int = 0) -> list[str]:
    if stage.GetPrimAtPath(material_root_prim_path).IsValid():
        custom_data = get_custom_data_from_prim(stage, material_root_prim_path)
        if "material_prim_paths" in custom_data:
            return custom_data["material_prim_paths"]
        stage.RemovePrim(material_root_prim_path)

    material_prim_paths = []
    material_count = 0
    for material_path in material_list:
        subidentifiers = asyncio.run(omni.kit.material.library.get_subidentifier_from_mdl(material_path))
        if not subidentifiers:
            print(f"Warning: No subidentifiers found for material: {material_path}, skipping.")
            continue
        for subidentifier in subidentifiers:
                    for i in range(random_transform_each_num):
                        material_prim_path = os.path.join(material_root_prim_path, f"RandomMaterial_{material_count}")
                        success, result = omni.kit.commands.execute("CreateMdlMaterialPrimCommand",
                            mtl_url=str(material_path),
                            mtl_name=str(subidentifier),
                            mtl_path=material_prim_path
                        )
                        if not success:
                            print(f"Failed to create material prim at path: {material_prim_path}")
                            continue
                        shader_prim = stage.GetPrimAtPath(os.path.join(material_prim_path, "Shader"))
                        if shader_prim.IsValid():
                            shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
                            shader_prim.CreateAttribute("inputs:world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
                            shader_prim.CreateAttribute("inputs:texture_rotate", Sdf.ValueTypeNames.Float).Set(np.random.uniform(0, 360))
                            shader_prim.CreateAttribute("inputs:texture_translate", Sdf.ValueTypeNames.Float2).Set((np.random.uniform(0, 100), np.random.uniform(0, 100)))

                        material_count += 1
                        material_prim_paths.append(material_prim_path)
    if random_color_material_list is not None:
        for material_path in random_color_material_list:
            subidentifiers = asyncio.run(omni.kit.material.library.get_subidentifier_from_mdl(material_path))
            if not subidentifiers:
                print(f"Warning: No subidentifiers found for material: {material_path}, skipping.")
                continue
            for i in range(random_color_each_num):
                material_prim_path = os.path.join(material_root_prim_path, f"RandomMaterial_{material_count}")
                success, result = omni.kit.commands.execute("CreateMdlMaterialPrimCommand",
                    mtl_url=str(material_path),
                    mtl_name=str(np.random.choice(subidentifiers)),
                    mtl_path=material_prim_path
                )
                if not success:
                    print(f"Failed to create material prim at path: {material_prim_path}")
                    continue
                shader_prim = stage.GetPrimAtPath(os.path.join(material_prim_path, "Shader"))
                if shader_prim.IsValid():
                    shader_prim.CreateAttribute("inputs:project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
                    shader_prim.CreateAttribute("inputs:texture_rotate", Sdf.ValueTypeNames.Float).Set(np.random.uniform(0, 360))
                    shader_prim.CreateAttribute("inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*np.random.exponential(0.02, 3)))
                material_count += 1
                material_prim_paths.append(material_prim_path)
    if material_prim_paths:
        write_custom_data_to_prim(stage, material_root_prim_path, {
            "material_prim_paths": material_prim_paths
        })
    else:
        print(f"Warning: No materials were loaded for {material_root_prim_path}. "
              "Material randomization will be skipped. This may happen if Omniverse Nucleus "
              "materials are not accessible.")
    return material_prim_paths


def add_rigid_body(stage: Usd.Stage, prim_path: str, kinematic: bool = False):
    prim = stage.GetPrimAtPath(prim_path)
    rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigidBodyAPI.CreateKinematicEnabledAttr(kinematic)


def add_mass(stage: Usd.Stage, prim_path: str, mass: float = 1.0):
    prim = stage.GetPrimAtPath(prim_path)
    massAPI = UsdPhysics.MassAPI.Apply(prim)
    massAPI.CreateMassAttr().Set(mass)


def add_collider(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
    collisionAPI.GetCollisionEnabledAttr().Set(True)


def create_prim(prim_path: str, prim_type: str):
    omni.kit.commands.execute("CreatePrimWithDefaultXform", prim_path=prim_path, prim_type=prim_type)


def create_plane(stage: Usd.Stage, prim_path: str, size: tuple[float, float]):
    create_prim(prim_path, "Mesh")
    plane_prim: UsdGeom.Mesh = UsdGeom.Mesh.Define(stage, prim_path)
    normal = Gf.Vec3f(0, 0, 1)
    points = [
        Gf.Vec3f(-size[0] / 2, -size[1] / 2, 0),
        Gf.Vec3f(size[0] / 2, -size[1] / 2, 0),
        Gf.Vec3f(size[0] / 2, size[1] / 2, 0),
        Gf.Vec3f(-size[0] / 2, size[1] / 2, 0)
    ]
    plane_prim.CreatePointsAttr(points)
    plane_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    plane_prim.CreateFaceVertexCountsAttr([4])
    plane_prim.CreateNormalsAttr([normal])

    return plane_prim


def create_rect_light(stage: Usd.Stage, prim_path: str, size: tuple[float, float], temperature: float = 6500.0, intensity: float = 1000.0):
    create_prim(prim_path, "RectLight")
    rect_light_prim: UsdLux.RectLight = UsdLux.RectLight.Define(stage, prim_path)
    rect_light_prim.CreateEnableColorTemperatureAttr().Set(True)
    rect_light_prim.CreateColorTemperatureAttr().Set(temperature)
    rect_light_prim.CreateIntensityAttr().Set(intensity * 100.0)
    rect_light_prim.CreateWidthAttr().Set(size[0])
    rect_light_prim.CreateHeightAttr().Set(size[1])


def write_custom_data_to_prim(stage: Usd.Stage, prim_path: str, custom_data: dict):
    prim = stage.GetPrimAtPath(prim_path)
    existing_custom_data = get_custom_data_from_prim(stage, prim_path)
    existing_custom_data.update(custom_data)
    prim.GetPrim().SetMetadata("customData", existing_custom_data)


def get_custom_data_from_prim(stage: Usd.Stage, prim_path: str) -> dict:
    prim = stage.GetPrimAtPath(prim_path)
    if prim.GetPrim().HasMetadata("customData"):
        return prim.GetPrim().GetMetadata("customData")
    return {}
