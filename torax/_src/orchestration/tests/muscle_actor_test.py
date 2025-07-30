import imas
import pytest
import torax

# libmuscle and ymmsl are optional dependencies, so may not be installed
libmuscle = pytest.importorskip("libmuscle")
ymmsl = pytest.importorskip("ymmsl")

from torax._src.orchestration.run_muscle3 import main as torax_actor

# imas_core is required for IDS serialize, unfortunately this means we cannot run these
# tests in github Actions yet..
pytest.importorskip("imas_core")


def source_for_tests():
    """MUSCLE3 actor sending out imas data to test torax-m3 actor"""
    instance = libmuscle.Instance({ymmsl.Operator.O_F: ["equilibrium_out"]})
    while instance.reuse_instance():
        imas_filepath = instance.get_setting("imas_source")
        with imas.DBEntry(uri=imas_filepath, mode="r") as db:
            equilibrium_data = db.get(ids_name="equilibrium")

        msg_equilibrium_out = libmuscle.Message(
            0, data=equilibrium_data.serialize(), next_timestamp=1
        )
        instance.send("equilibrium_out", msg_equilibrium_out)


def sink_for_tests():
    """MUSCLE3 actor receiving imas data to test torax-m3 actor"""
    instance = libmuscle.Instance({ymmsl.Operator.F_INIT: ["equilibrium_in"]})
    while instance.reuse_instance():
        data_sink_path = instance.get_setting("imas_sink")
        msg_equilibrium_in = instance.receive("equilibrium_in")
        equilibrium_data = imas.IDSFactory().equilibrium()
        equilibrium_data.deserialize(msg_equilibrium_in.data)
        with imas.DBEntry(uri=data_sink_path, mode="w") as db:
            db.put(equilibrium_data)


YMMSL_OUTPUT = """
ymmsl_version: v0.1
model:
  name: test_model
  components:
    sink:
      implementation: sink
      ports:
        f_init: [equilibrium_in]
    torax:
      implementation: torax
      ports:
        o_f: [equilibrium_o_f]
  conduits:
    torax.equilibrium_o_f: sink.equilibrium_in
settings:
  sink.imas_sink: {data_sink_path}
  torax.python_config_module: {config_path}
"""

YMMSL_INPUT = """
ymmsl_version: v0.1
model:
  name: test_model
  components:
    source:
      implementation: source
      ports:
        o_f: [equilibrium_out]
    torax:
      implementation: torax
      ports:
        f_init: [equilibrium_f_init]
  conduits:
    source.equilibrium_out: torax.equilibrium_f_init
settings:
  source.imas_source: {data_source_path}
  torax.python_config_module: {config_path}
"""


# Running `os.fork()` after `import pandas` triggers this warning...
# It doesn't seem to be an issue (and not relevant in production where muscle_manager
# will start the actor in a standalone process), so we'll ignore this warning:
@pytest.mark.filterwarnings("ignore:.*use of fork():DeprecationWarning")
def test_actor_input(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filename = "ITERhybrid_COCOS17_IDS_ddv4.nc"
    data_source_path = f"{torax.__path__[0]}/data/third_party/geo/{filename}"
    # config_path = f"{torax.__path__[0]}/examples/iterhybrid_predictor_corrector.py"
    config_path = f"{torax.__path__[0]}/examples/basic_config.py"
    configuration = ymmsl.load(
        YMMSL_INPUT.format(
            data_source_path=data_source_path,
            config_path=config_path,
        )
    )
    implementations = {
        "source": source_for_tests,
        "torax": torax_actor,
    }
    libmuscle.runner.run_simulation(configuration, implementations)


@pytest.mark.filterwarnings("ignore:.*use of fork():DeprecationWarning")
def test_actor_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_sink_path = (tmp_path / "sink_dir").absolute()
    config_path = f"{torax.__path__[0]}/examples/iterhybrid_predictor_corrector.py"
    configuration = ymmsl.load(
        YMMSL_OUTPUT.format(
            data_sink_path=data_sink_path,
            config_path=config_path,
        )
    )
    implementations = {
        "sink": sink_for_tests,
        "torax": torax_actor,
    }
    libmuscle.runner.run_simulation(configuration, implementations)

