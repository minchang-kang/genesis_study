from dataclasses import MISSING
import genesis as gs

from .from_file_cfg import *
from genesis_study.source.genesis.utils import configclass


@configclass
class EntityAssetCfg:
    morph: URDFCfg = MISSING
    material: gs.materials.Rigid | None = None
    surface: gs.surfaces.Surface | None = None
    visualize_contact: bool = False
    vis_mode: str | None= None


@configclass
class LightAssetCfg:
    morph: URDFCfg = MISSING
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 20.0
    revert_dir: bool = False
    double_sided: bool = False
    beam_angle: float = 180.0


@configclass
class CameraAssetCfg:
    model: str = 'pinhole'
    res: tuple[int, int] = (320, 320)
    pos:tuple[float, float, float] = (0.5, 2.5, 3.5)
    lookat: tuple[float, float, float] = (0.5, 0.5, 0.5)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 30
    aperture: float = 2.0
    focus_dist: float | None = None
    GUI: bool = False
    spp: int = 256
    denoise: bool = True


@configclass
class FluidEmitterAssetCfg:
    material: gs.materials = MISSING
    max_particles: int = 20000
    surface: gs.surfaces.Surface | None = None
