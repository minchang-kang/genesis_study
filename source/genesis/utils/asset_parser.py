import 






def parse_morph(cfg: Any):
    if isinstance(cfg, URDFCfg):
        return gs.morphs.URDF(
            file=cfg.file,
            pos=cfg.pos,
            quat=cfg.quat,
            euler=cfg.euler,
            scale=cfg.scale,
            fixed=cfg.fixed,
            convexify=cfg.convexify,
            visualization=cfg.visualization,
            collision=cfg.colision,
            requires_jacobian_and_ik=cfg.requires_jac_and_IK,
            prioritize_urdf_material=cfg.prioritize__urdf_material,
            merge_fixed_links=cfg.merge_fixed_links,
            links_to_keep=cfg.links_to_keep,
        )
    raise ValueError(f"[parse_morph] Unknown morph config type: {type(cfg)}")

def parse_material(cfg: Any):
    if cfg is None:
        return None
    if isinstance(cfg, MaterialsCfg):
        return gs.materials.Rigid(rho=cfg.rho)
    raise ValueError(f"[parse_material] Unknown material config type: {type(cfg)}")

def parse_surface(cfg: Any):
    if cfg is None:
        return None
    if isinstance(cfg, SurfaceCfg):
        return gs.surfaces.Default(
            color=cfg.color,
            vis_mode=cfg.vis_mode,
        )
    raise ValueError(f"[parse_surface] Unknown surface config type: {type(cfg)}")

def _create_scene_entities(scene, name, asset_cfg):
    """Scene에 엔티티 자동 추가."""

    if isinstance(self.cfg.scene, dict):
        scene_items = self.cfg.scene.items()
    else:
        scene_items = self.cfg.scene.__dict__.items()

    handler_map = {
        EntityAssetCfg: lambda cfg: self.scene.add_entity(
            morph=parse_morph(cfg.morph),
            material=parse_material(cfg.material),
            surface=parse_surface(cfg.surface),
            visualize_contact=cfg.visualize_contact,
            vis_mode=cfg.vis_mode,
        ),
        LightAssetCfg: lambda cfg: self.scene.add_light(
            morph=parse_morph(cfg.morph),
            color=cfg.color,
            intensity=cfg.intensity,
            revert_dir=cfg.revert_dir,
            double_sided=cfg.double_sided,
            beam_angle=cfg.beam_angle,
        ),
        CameraAssetCfg: lambda cfg: self.scene.add_camera(
            model=cfg.model,
            res=cfg.res,
            pos=cfg.pos,
            lookat=cfg.lookat,
            up=cfg.up,
            fov=cfg.fov,
            aperture=cfg.aperture,
            focus_dist=cfg.focus_dist,
            GUI=cfg.GUI,
            spp=cfg.spp,
            denoise=cfg.denoise,
        ),
        FluidEmitterAssetCfg: lambda cfg: self.scene.add_emitter(
            material=parse_material(cfg.material),
            max_particles=cfg.max_particles,
            surface=parse_surface(cfg.surface),
        ),
    }

    for name, asset_cfg in scene_items:
        if asset_cfg is None:
            continue
        for asset_type, handler in handler_map.items():
            if isinstance(asset_cfg, asset_type):
                entity = handler(asset_cfg)
                setattr(self, name, entity)
                break
