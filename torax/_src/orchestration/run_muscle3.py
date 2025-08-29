r"""
MUSCLE3 actor wrapping TORAX.

Configuration can be specified as a path to a config file,.
and individual muscle3 config keys will be overwritten on that.

Start without inputs and outputs, and then add a static and
later dynamic equilibrium input.

Last (for sure) compatible torax commit: 4b76ef0566
"""

import logging

from imas import DBEntry
from libmuscle import Instance
from torax._src.config import build_runtime_params
from torax._src.config.build_runtime_params import get_consistent_dynamic_runtime_params_slice_and_geometry
from torax._src.config.config_loader import build_torax_config_from_file
from torax._src.orchestration import initial_state as initial_state_lib
from torax._src.orchestration.muscle3_utils import get_geometry_config_dict
from torax._src.orchestration.muscle3_utils import get_setting_optional
from torax._src.orchestration.muscle3_utils import get_t_next
from torax._src.orchestration.muscle3_utils import merge_extra_vars
from torax._src.orchestration.muscle3_utils import receive_equilibrium
from torax._src.orchestration.muscle3_utils import receive_ids_through_muscle3
from torax._src.orchestration.muscle3_utils import send_ids_through_muscle3
from torax._src.orchestration.run_simulation import prepare_simulation
from torax._src.output_tools.imas import torax_state_to_imas_equilibrium
from torax._src.state import SimError
from ymmsl import Operator

logger = logging.getLogger()


def main() -> None:
    """Create TORAX instance and enter submodel execution loop"""
    logger.info("Starting TORAX actor")
    instance = Instance(
        {
            Operator.F_INIT: ["equilibrium_f_init"],
            Operator.O_I: ["equilibrium_o_i"],
            Operator.S: ["equilibrium_s"],
            Operator.O_F: ["equilibrium_o_f"],
        }
    )

    # enter re-use loop
    first_run = True
    while instance.reuse_instance():
        if first_run:
            # load config file from path
            config_module_str = instance.get_setting("python_config_module")
            output_all_timeslices = get_setting_optional(
                instance, "output_all_timeslices", False
            )
            if output_all_timeslices:
                db_out = DBEntry("imas:memory?path=/db_out/", "w")

            torax_config = build_torax_config_from_file(
                path=config_module_str,
            )
            (
                dynamic_runtime_params_slice_provider,
                sim_state,
                post_processed_outputs,
                step_fn,
            ) = prepare_simulation(torax_config)

            time_step_calculator_dynamic_params = dynamic_runtime_params_slice_provider(
                sim_state.t
            ).time_step_calculator
            extra_var_col = None
            t_next_outer = None

        # F_INIT
        if instance.is_connected("equilibrium_f_init"):
            step_fn._geometry_provider, t_cur, t_next_outer, extra_var_col = receive_equilibrium(
                instance,
                step_fn._geometry_provider,
                sim_state,
                torax_config,
                "equilibrium_f_init",
            )
            sim_state.t = t_cur
        if first_run or instance.is_connected("equilibrium_f_init"):
            static_runtime_params_slice = (
                build_runtime_params.build_static_params_from_config(torax_config)
            )
            dynamic_runtime_params_slice, _ = (
                get_consistent_dynamic_runtime_params_slice_and_geometry(
                    t=sim_state.t,
                    dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
                    geometry_provider=step_fn._geometry_provider,
                )
            )
            # next function loses sim_state.t information so readd it
            my_t = sim_state.t
            sim_state, post_processed_outputs = (
                initial_state_lib.get_initial_state_and_post_processed_outputs(
                    t=sim_state.t,
                    static_runtime_params_slice=static_runtime_params_slice,
                    dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
                    geometry_provider=step_fn._geometry_provider,
                    step_fn=step_fn,
                )
            )
            sim_state.t = my_t
            sim_state.geometry = step_fn._geometry_provider(sim_state.t)
        first_run = False

        # temp
        sim_state, post_processed_outputs, sim_error = step_fn(
            sim_state,
            post_processed_outputs,
        )
        # temp

        equilibrium_data = torax_state_to_imas_equilibrium(sim_state, post_processed_outputs)
        if extra_var_col is not None:
            equilibrium_data = merge_extra_vars(equilibrium_data, extra_var_col)

        if output_all_timeslices:
            db_out.put_slice(equilibrium_data)

        # TODO: Needs proper way to implement stopping condition if used as micro component
        # Advance the simulation until the time_step_calculator tells us we are done.
        while step_fn.time_step_calculator.not_done(
            sim_state.t,
            dynamic_runtime_params_slice.numerics.t_final,
            time_step_calculator_dynamic_params,
        ):
            # O_I
            t_next_inner = get_t_next(
                sim_state,
                dynamic_runtime_params_slice_provider,
                dynamic_runtime_params_slice,
                step_fn,
                step_fn._geometry_provider,
            )
            send_ids_through_muscle3(
                instance,
                equilibrium_data,
                "equilibrium_o_i",
                sim_state.t,
                t_next=t_next_inner,
            )

            # S
            if instance.is_connected("equilibrium_s"):
                step_fn._geometry_provider, t_cur, _, extra_var_col = receive_equilibrium(
                    instance,
                    step_fn._geometry_provider,
                    sim_state,
                    torax_config,
                    "equilibrium_s",
                )

            sim_state, post_processed_outputs, sim_error = step_fn(
                sim_state,
                post_processed_outputs,
            )
            equilibrium_data = torax_state_to_imas_equilibrium(sim_state, post_processed_outputs)
            if extra_var_col is not None:
                equilibrium_data = merge_extra_vars(equilibrium_data, extra_var_col)

            if output_all_timeslices:
                db_out.put_slice(equilibrium_data)

            if sim_error != SimError.NO_ERROR:
                raise Exception()

        # Update the final time step's source profiles based on the explicit source
        # profiles computed based on the final state.
        logging.info("Updating last step's source profiles.")

        # O_F
        if output_all_timeslices:
            equilibrium_data = db_out.get("equilibrium")
            db_out.close()

        send_ids_through_muscle3(
            instance,
            equilibrium_data,
            "equilibrium_o_f",
            sim_state.t,
            t_next=t_next_outer,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
