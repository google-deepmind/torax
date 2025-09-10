r"""
MUSCLE3 actor wrapping TORAX.

Configuration can be specified as a path to a config file,.
and individual muscle3 config keys will be overwritten on that.

Start without inputs and outputs, and then add a static and
later dynamic equilibrium input.

Last (for sure) compatible torax commit: 4b76ef0566
"""

import logging
from typing import Optional

from imas import DBEntry
from imas import IDSFactory
from imas.ids_defs import CLOSEST_INTERP
from imas.ids_toplevel import IDSToplevel
from libmuscle import Instance
from libmuscle import Message
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.config.build_runtime_params import get_consistent_dynamic_runtime_params_slice_and_geometry
from torax._src.config.config_loader import build_torax_config_from_file
from torax._src.geometry import geometry
from torax._src.geometry.pydantic_model import Geometry
from torax._src.geometry.pydantic_model import GeometryConfig
from torax._src.geometry.pydantic_model import IMASConfig
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration import sim_state
from torax._src.orchestration.muscle3_utils import ExtraVarCollection
from torax._src.orchestration.muscle3_utils import get_geometry_config_dict
from torax._src.orchestration.muscle3_utils import get_setting_optional
from torax._src.orchestration.muscle3_utils import merge_extra_vars
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.output_tools.imas import torax_state_to_imas_equilibrium
from torax._src.state import SimError
from ymmsl import Operator

logger = logging.getLogger()


class ToraxMuscleRunner:
    first_run: bool = True
    output_all_timeslices: bool = False
    db_out: Optional[IDSToplevel] = None
    torax_config = None
    dynamic_runtime_params_slice_provider = None
    sim_state = None
    post_processed_outputs = None
    step_fn = None
    time_step_calculator_dynamic_params = None
    extra_var_col = None
    t_cur = None
    t_next_inner = None
    t_next_outer = None
    finished = False

    def __init__(self):
        self.get_instance()
        self.extra_var_col = ExtraVarCollection()

    def run_sim(self):
        if self.finished:
            raise Warning("Already finished")

        while self.instance.reuse_instance():
            if self.first_run:
                self.run_prep()
            self.run_f_init()
            while self.step_fn.time_step_calculator.not_done(
                self.t_cur,
                self.t_final,
                self.time_step_calculator_dynamic_params,
            ):
                self.run_o_i()
                self.run_s()
                self.run_timestep()
            self.run_o_f()

        self.finished = True

    def run_prep(self):
        self.output_all_timeslices = get_setting_optional(
            self.instance, "output_all_timeslices", False
        )
        # load config file from path
        config_module_str = self.instance.get_setting("python_config_module")
        if self.output_all_timeslices:
            self.db_out = DBEntry("imas:memory?path=/db_out/", "w")
        self.torax_config = build_torax_config_from_file(
            path=config_module_str,
        )
        (
            self.dynamic_runtime_params_slice_provider,
            self.sim_state,
            self.post_processed_outputs,
            self.step_fn,
        ) = prepare_simulation(self.torax_config)

        self.time_step_calculator_dynamic_params = (
            self.dynamic_runtime_params_slice_provider(
                self.sim_state.t
            ).time_step_calculator
        )

    def run_f_init(self):
        self.receive_equilibrium(port_name="f_init")
        self.sim_state.t = self.t_cur
        if self.first_run or self.instance.is_connected("equilibrium_f_init"):
            static_runtime_params_slice = (
                build_runtime_params.build_static_params_from_config(self.torax_config)
            )
            dynamic_runtime_params_slice, _ = (
                get_consistent_dynamic_runtime_params_slice_and_geometry(
                    t=self.sim_state.t,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.step_fn._geometry_provider,
                )
            )
            # next function loses sim_state.t information so readd it
            self.t_cur = self.sim_state.t
            self.sim_state, self.post_processed_outputs = (
                initial_state_lib.get_initial_state_and_post_processed_outputs(
                    t=self.sim_state.t,
                    static_runtime_params_slice=static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                    geometry_provider=self.step_fn._geometry_provider,
                    step_fn=self.step_fn,
                )
            )
            self.sim_state.t = self.t_cur
            self.sim_state.geometry = self.step_fn._geometry_provider(self.sim_state.t)
            self.t_final = dynamic_runtime_params_slice.numerics.t_final
        self.first_run = False

        equilibrium_data = torax_state_to_imas_equilibrium(
            self.sim_state, self.post_processed_outputs
        )
        if self.extra_var_col is not None:
            equilibrium_data = merge_extra_vars(equilibrium_data, self.extra_var_col)

        if self.output_all_timeslices:
            self.db_out.put_slice(equilibrium_data)

    def run_o_i(self):
        self.t_next_inner = self.get_t_next()
        if self.instance.is_connected("equilibrium_o_i"):
            equilibrium_data = torax_state_to_imas_equilibrium(
                self.sim_state, self.post_processed_outputs
            )
            if self.extra_var_col is not None:
                equilibrium_data = merge_extra_vars(
                    equilibrium_data, self.extra_var_col
                )
            self.send_ids(equilibrium_data, "equilibrium", "o_i")

    def run_s(self):
        if self.instance.is_connected("equilibrium_s"):
            self.receive_equilibrium(port_name="s")

    def run_timestep(self):
        self.sim_state, self.post_processed_outputs, sim_error = self.step_fn(
            self.sim_state,
            self.post_processed_outputs,
        )

        if self.output_all_timeslices:
            equilibrium_data = torax_state_to_imas_equilibrium(
                self.sim_state, self.post_processed_outputs
            )
            if self.extra_var_col is not None:
                equilibrium_data = merge_extra_vars(
                    equilibrium_data, self.extra_var_col
                )
            self.db_out.put_slice(equilibrium_data)

        if sim_error != SimError.NO_ERROR:
            raise Exception()

    def run_o_f(self):
        if self.output_all_timeslices:
            equilibrium_data = self.db_out.get("equilibrium")
            self.db_out.close()

        self.send_ids(equilibrium_data, "equilibrium", "o_f")

    def get_instance(self):
        self.instance = Instance(
            {
                Operator.F_INIT: ["equilibrium_f_init"],
                Operator.O_I: ["equilibrium_o_i"],
                Operator.S: ["equilibrium_s"],
                Operator.O_F: ["equilibrium_o_f"],
            }
        )

    def receive_equilibrium(self, port_name: str):
        if not self.instance.is_connected(f"equilibrium_{port_name}"):
            return
        equilibrium_data, self.t_cur, t_next = self.receive_ids(
            "equilibrium", port_name
        )
        if port_name == "f_init":
            self.t_next_outer = t_next
        elif port_name == "s":
            self.t_next_inner = t_next
        geometry_configs = {}
        torax_config_dict = get_geometry_config_dict(self.torax_config)
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
                self.extra_var_col.add_val(
                    "z_boundary_outline",
                    t,
                    np.asarray(my_slice.time_slice[0].boundary.outline.z),
                )
                self.extra_var_col.add_val(
                    "r_boundary_outline",
                    t,
                    np.asarray(my_slice.time_slice[0].boundary.outline.r),
                )
        self.step_fn._geometry_provider = Geometry(
            geometry_type=geometry.GeometryType.IMAS,
            geometry_configs=geometry_configs,
        ).build_provider
        # temp extra vars code
        self.extra_var_col.pad_extra_vars()

    def receive_ids(self, ids_name, port_name):
        if not self.instance.is_connected(f"{ids_name}_{port_name}"):
            return
        msg = self.instance.receive(f"{ids_name}_{port_name}")
        t_cur = msg.timestamp
        t_next = msg.next_timestamp
        ids_data = getattr(IDSFactory(), ids_name)()
        ids_data.deserialize(msg.data)
        return ids_data, t_cur, t_next

    def send_ids(self, ids, ids_name, port_name):
        if not self.instance.is_connected(f"{ids_name}_{port_name}"):
            return
        if port_name == "o_i":
            t_next = self.t_next_inner
        elif port_name == "o_f":
            t_next = self.t_next_outer
        msg = Message(self.t_cur, data=ids.serialize(), next_timestamp=t_next)
        self.instance.send(f"{ids_name}_{port_name}", msg)

    def get_t_next(self):
        dynamic_runtime_params_slice_t, geo_t = (
            get_consistent_dynamic_runtime_params_slice_and_geometry(
                t=self.sim_state.t,
                dynamic_runtime_params_slice_provider=self.dynamic_runtime_params_slice_provider,
                geometry_provider=self.step_fn._geometry_provider,
            )
        )
        dt = self.step_fn.time_step_calculator.next_dt(
            self.sim_state.t,
            dynamic_runtime_params_slice_t,
            geo_t,
            self.sim_state.core_profiles,
            self.sim_state.core_transport,
        )
        t_next = sim_state.t + dt
        if t_next > self.t_final:
            t_next = None
        return t_next


def main() -> None:
    """Create TORAX instance and enter submodel execution loop"""
    logger.info("Starting TORAX actor")
    tmr = ToraxMuscleRunner()
    tmr.run_sim()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
