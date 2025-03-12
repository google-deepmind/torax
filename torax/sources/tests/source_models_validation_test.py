

"""Tests for validation checks in source_models.py."""

import pytest
from torax.sources import bremsstrahlung_heat_sink
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit


def test_bremsstrahlung_and_mavrin_active_check():
  """Tests that bremsstrahlung and mavrin impurity radiation models can't both be active."""

  builder = source_models.SourceModelsBuilder({
      bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME:
          bremsstrahlung_heat_sink.BremsstrahlungHeatSink(
              model_func=bremsstrahlung_heat_sink.bremsstrahlung_model_func,
          ),
      impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME:
          impurity_radiation_heat_sink.ImpurityRadiationHeatSink(
              model_func=impurity_radiation_mavrin_fit.impurity_radiation_mavrin_fit,
          ),
  })
  

  builder.runtime_params[bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.ZERO
  builder.runtime_params[impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.ZERO
  _ = builder()  
  

  builder.runtime_params[bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.MODEL_BASED
  builder.runtime_params[impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.ZERO
  _ = builder()  
  
  
  builder.runtime_params[bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.ZERO
  builder.runtime_params[impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.MODEL_BASED
  _ = builder()  

  builder.runtime_params[bremsstrahlung_heat_sink.BremsstrahlungHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.MODEL_BASED
  builder.runtime_params[impurity_radiation_heat_sink.ImpurityRadiationHeatSink.SOURCE_NAME].mode = runtime_params_lib.Mode.MODEL_BASED
  
  with pytest.raises(ValueError) as excinfo:
    _ = builder()
    
  
  assert "Both bremsstrahlung_heat_sink and impurity_radiation_heat_sink with Mavrin model" in str(excinfo.value)
  assert "should not be active at the same time" in str(excinfo.value) 