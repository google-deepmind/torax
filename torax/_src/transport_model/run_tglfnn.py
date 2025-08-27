import torax
from torax import plotting
from torax._src.config import config_loader
from torax._src.plotting.plotruns_lib import plot_run
from torax.tests.test_data import test_iterhybrid_predictor_corrector_reference
from torax.tests.test_data import test_iterhybrid_predictor_corrector_tglfnn_ukaea


def run_and_save(config_dict, outfile):
  config = torax.ToraxConfig.from_dict(config_dict)
  data_tree, state_history = torax.run_simulation(config)
  data_tree.to_netcdf(outfile)


run_and_save(
    test_iterhybrid_predictor_corrector_tglfnn_ukaea.CONFIG, "tglfnn_ukaea.nc"
)
run_and_save(
    test_iterhybrid_predictor_corrector_reference.CONFIG, "reference.nc"
)

PLOT_CONFIG = config_loader.import_module(
    "plotting/configs/default_plot_config.py"
)["PLOT_CONFIG"]

plot_run(
    PLOT_CONFIG,
    "tglfnn_ukaea.nc",
    "reference.nc",
)
