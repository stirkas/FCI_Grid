from dataclasses import dataclass
from typing import Any

from grid import StructuredGrid
from field import MagneticField
from boundary import PolygonBoundary

#TODO: Find a better place for this, here for now to break dependencies.
@dataclass(slots=True)
class BOUT_IO:
    """Derived output maps ready for IO."""
    maps: dict
    metric: dict
    attributes: dict

@dataclass(slots=True)
class DeviceInfo:
    """Device-specific information which affects generation logic."""
    axisymmetric: bool = False
    toroidal: bool = False

@dataclass(slots=True)
class Device:
    """Raw device components."""
    data:  Any #Don't really have a generic way to structure specific data.
    grid:  StructuredGrid
    field: MagneticField
    wall:  PolygonBoundary
    dvc_info: DeviceInfo