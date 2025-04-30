from typing import Any
import genesis as gs
from genesis_study.source.genesis.assets.asset_base_cfg import *
from genesis_study.source.genesis.assets.material_cfg import *
from genesis_study.source.genesis.assets.surface_cfg import *


def parse_morph(cfg):
    if isinstance(cfg, URDFCfg):
        # quat = cfg.quat if cfg.quat is not None else quat_from_euler(cfg.euler)
        return gs.morphs.URDF(
            file=cfg.file,
            scale=cfg.scale,
            pos=cfg.pos,
            quat=cfg.quat,
            fixed=cfg.fixed,
        )
    elif isinstance(cfg, BoxCfg):
        # quat = cfg.quat if cfg.quat is not None else quat_from_euler(cfg.euler)
        return gs.morphs.Box(
            size=cfg.size,
            pos=cfg.pos,
            quat=cfg.quat,
        )
    elif isinstance(cfg, MJCFCfg):
        return gs.morphs.MJCF(
            file=cfg.file,
            scale=cfg.scale,
            pos=cfg.pos,
            quat=cfg.quat,
        )
    else:
        raise TypeError(f"Unknown morph config type: {type(cfg)}")


def parse_material(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, MaterialsCfg):
        return gs.materials.Rigid(rho=cfg.rho)
    raise TypeError(f"Unknown material config type: {type(cfg)}")


def parse_surface(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, SurfaceCfg):
        return gs.surfaces.Default(color=cfg.color, vis_mode=cfg.vis_mode)
    raise TypeError(f"Unknown surface config type: {type(cfg)}")


def create_scene_entity_from_cfg(scene, name, asset_cfg):
    if isinstance(asset_cfg, EntityAssetCfg):
        morph = parse_morph(asset_cfg.morph)
        return scene.add_entity(
            morph,
            material=parse_material(asset_cfg.material),
            surface=parse_surface(asset_cfg.surface),
            visualize_contact=asset_cfg.visualize_contact,
            vis_mode=asset_cfg.vis_mode,
        )

    elif isinstance(asset_cfg, LightAssetCfg):
        morph = parse_morph(asset_cfg.morph)
        return scene.add_light(
            morph,
            color=asset_cfg.color,
            intensity=asset_cfg.intensity,
            revert_dir=asset_cfg.revert_dir,
            double_sided=asset_cfg.double_sided,
            beam_angle=asset_cfg.beam_angle,
        )

    elif isinstance(asset_cfg, CameraAssetCfg):
        return scene.add_camera(
            model=asset_cfg.model,
            res=asset_cfg.res,
            pos=asset_cfg.pos,
            lookat=asset_cfg.lookat,
            up=asset_cfg.up,
            fov=asset_cfg.fov,
            aperture=asset_cfg.aperture,
            focus_dist=asset_cfg.focus_dist,
            GUI=asset_cfg.GUI,
            spp=asset_cfg.spp,
            denoise=asset_cfg.denoise,
        )

    elif isinstance(asset_cfg, FluidEmitterAssetCfg):
        return scene.add_emitter(
            material=parse_material(asset_cfg.material),
            max_particles=asset_cfg.max_particles,
            surface=parse_surface(asset_cfg.surface),
        )

    return None
