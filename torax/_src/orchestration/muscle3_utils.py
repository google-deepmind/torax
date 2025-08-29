"""
Utility functions for muscle3 and torax.
"""

import logging
from typing import Optional

from imas import DBEntry
from libmuscle import Instance
from torax._src.config.build_runtime_params import DynamicRuntimeParamsSliceProvider
from torax._src.config.build_runtime_params import get_consistent_dynamic_runtime_params_slice_and_geometry
from torax._src.config.config_loader import build_torax_config_from_file
from torax._src.config.runtime_params_slice import DynamicRuntimeParamsSlice
from torax._src.geometry.geometry_provider import GeometryProvider
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.output_tools.imas import torax_state_to_imas_equilibrium
from torax._src.state import SimError
from ymmsl import Operator
from ymmsl import SettingValue

logger = logging.getLogger()


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from imas import DBEntry
from imas import IDSFactory
from imas.ids_defs import CLOSEST_INTERP
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance
from libmuscle import Message
import numpy as np
from torax._src.geometry import geometry
from torax._src.geometry.geometry_provider import GeometryProvider
from torax._src.geometry.pydantic_model import Geometry
from torax._src.geometry.pydantic_model import GeometryConfig
from torax._src.geometry.pydantic_model import IMASConfig
from torax._src.orchestration import sim_state
from torax._src.torax_pydantic import model_config


@dataclass
class ExtraVarDir:
    """Temp code for extra vars"""

    name: str
    xs: List[float]
    ys: List


class ExtraVarCollection:
    """Temp code for extra vars"""

    extra_var_dirs: Dict[str, ExtraVarDir]

    def __init__(self, names: List[str] = []) -> None:
        self.extra_var_dirs = {
            name: ExtraVarDir(name=name, xs=[], ys=[]) for name in names
        }

    def add_val(self, name: str, x: float, y: Any) -> None:
        if name not in self.extra_var_dirs.keys():
            self.extra_var_dirs[name] = ExtraVarDir(name=name, xs=[], ys=[])
        self.extra_var_dirs[name].xs.append(x)
        self.extra_var_dirs[name].ys.append(y)

    def pad_extra_vars(self) -> None:
        for name in self.extra_var_dirs.keys():
            self.extra_var_dirs[name].xs = (
                [-np.inf] + self.extra_var_dirs[name].xs + [np.inf]
            )
            self.extra_var_dirs[name].ys = (
                [self.extra_var_dirs[name].ys[0]]
                + self.extra_var_dirs[name].ys
                + [self.extra_var_dirs[name].ys[-1]]
            )

    def get_val(self, name: str, x: float) -> Any:
        """Step interpolation"""
        var_dir = self.extra_var_dirs[name]
        idx = max(i for i in range(len(var_dir.xs)) if var_dir.xs[i] <= x)
        return var_dir.ys[idx]


def receive_equilibrium(
    instance: Instance,
    geometry_provider: GeometryProvider,
    state: sim_state.ToraxSimState,
    config: model_config.ToraxConfig,
    port_name: str,
) -> Tuple[GeometryProvider, float, Optional[float], ExtraVarCollection]:
    # TODO: receive all inputs [equilibrium, core_profiles, core_profiles]
    if not instance.is_connected(port_name):
        return geometry_provider, state.t, None, ExtraVarCollection()
    equilibrium_data, t_cur, t_next = receive_ids_through_muscle3(
        instance, port_name, 'equilibrium'
    )

    geometry_configs = {}
    extra_var_col = ExtraVarCollection()
    torax_config_dict = get_geometry_config_dict(config)
    torax_config_dict["geometry_type"] = "imas"
    with DBEntry("imas:memory?path=/", "w") as db:
        db.put(equilibrium_data)
        for t in equilibrium_data.time:
            my_slice = db.get_slice(
                ids_name="equilibrium",
                time_requested=t,
                interpolation_method=CLOSEST_INTERP,
            )
            config_kwargs = {
                **torax_config_dict,
                "equilibrium_object": my_slice,
                "imas_uri": None,
                "imas_filepath": None,
                "Ip_from_parameters": False,
            }
            imas_cfg = IMASConfig(**config_kwargs)
            cfg = GeometryConfig(config=imas_cfg)
            geometry_configs[str(t)] = cfg
            # temp extra vars code
            extra_var_col.add_val(
                "z_boundary_outline",
                t,
                np.asarray(my_slice.time_slice[0].boundary.outline.z),
            )
            extra_var_col.add_val(
                "r_boundary_outline",
                t,
                np.asarray(my_slice.time_slice[0].boundary.outline.r),
            )
    geometry_provider = Geometry(
        geometry_type=geometry.GeometryType.IMAS,
        geometry_configs=geometry_configs,
    ).build_provider
    # temp extra vars code
    extra_var_col.pad_extra_vars()
    return geometry_provider, t_cur, t_next, extra_var_col

def receive_ids_through_muscle3(
    instance: Instance,    
    port_name: str,
    ids_name: str,
) -> Tuple[IDSToplevel, float, Optional[float]]:
    msg_equilibrium_in = instance.receive(port_name)
    t_cur = msg_equilibrium_in.timestamp
    t_next = msg_equilibrium_in.next_timestamp
    equilibrium_data = getattr(IDSFactory(), "equilibrium")()
    equilibrium_data.deserialize(msg_equilibrium_in.data)
    return equilibrium_data, t_cur, t_next

def send_ids_through_muscle3(
    instance: Instance,
    equilibrium_data: IDSToplevel,
    port_name: str,
    t_cur: float,
    t_next: Optional[float] = None,
) -> None:
    if not instance.is_connected(port_name):
        return
    msg_equilibrium_out = Message(
        t_cur, data=equilibrium_data.serialize(), next_timestamp=t_next
    )
    instance.send(port_name, msg_equilibrium_out)


def get_geometry_config_dict(config: model_config.ToraxConfig) -> dict:
    # only get overlapping keys from given config and IMASConfig
    imas_config_keys = IMASConfig.__annotations__
    # we can pick a random entry since all fields are time_invariant except hires_fac
    # (which we can ignore) and equilibrium_object (which we overwrite)
    if isinstance(config.geometry.geometry_configs, dict):
        config_dict = list(config.geometry.geometry_configs.values())[0].config.__dict__
    else:
        config_dict = config.geometry.geometry_configs.config.__dict__
    config_dict = {
        key: value for key, value in config_dict.items() if key in imas_config_keys
    }
    return config_dict


def get_setting_optional(
    instance: Instance, setting_name: str, default: Optional[SettingValue] = None
) -> Optional[SettingValue]:
    """Helper function to get optional settings from instance"""
    setting: Optional[SettingValue]
    try:
        setting = instance.get_setting(setting_name)
    except KeyError:
        setting = default
    return setting


def get_t_next(
    sim_state: sim_state.ToraxSimState,
    dynamic_runtime_params_slice_provider: DynamicRuntimeParamsSliceProvider,
    dynamic_runtime_params_slice: DynamicRuntimeParamsSlice,
    step_fn: SimulationStepFn,
    geometry_provider: GeometryProvider,
) -> float:
    dynamic_runtime_params_slice_t, geo_t = (
        get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=sim_state.t,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=geometry_provider,
        )
    )
    dt = step_fn.time_step_calculator.next_dt(
        sim_state.t,
        dynamic_runtime_params_slice_t,
        geo_t,
        sim_state.core_profiles,
        sim_state.core_transport
    )
    t_next = sim_state.t + dt
    if t_next > dynamic_runtime_params_slice.numerics.t_final:
        t_next = None
    return t_next

def merge_extra_vars(equilibrium_data: IDSToplevel, extra_var_col: ExtraVarCollection):
    equilibrium_data.time_slice[0].boundary.outline.z = extra_var_col.get_val(
        "z_boundary_outline", equilibrium_data.time[0]
    )
    equilibrium_data.time_slice[0].boundary.outline.r = extra_var_col.get_val(
        "r_boundary_outline", equilibrium_data.time[0]
    )
    return equilibrium_data
