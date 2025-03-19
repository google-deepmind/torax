"""Tests for validation checks in source_models.py."""

import pytest

from torax.sources import bremsstrahlung_heat_sink
from torax.sources import runtime_params
from torax.sources import pydantic_model
from torax.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit


def test_bremsstrahlung_and_mavrin_active_check():
  """Tests that bremsstrahlung and mavrin impurity radiation models can't both be active simultaneously."""
  
  # Test with both active - should raise ValueError
  with pytest.raises(ValueError) as excinfo:
    # Direct dictionary input with both sources active
    config = {
        'bremsstrahlung_heat_sink': {
            'mode': 'MODEL_BASED',
            'source_name': 'bremsstrahlung_heat_sink',
        },
        'impurity_radiation_heat_sink': {
            'model_function_name': 'impurity_radiation_mavrin_fit',
            'mode': 'MODEL_BASED',
            'source_name': 'impurity_radiation_heat_sink',
        },
        'j_bootstrap': {
            'mode': 'ZERO',
            'source_name': 'j_bootstrap',
        },
        'generic_current_source': {
            'mode': 'ZERO',
            'source_name': 'generic_current_source',
        },
        'qei_source': {
            'mode': 'ZERO',
            'source_name': 'qei_source',
        },
    }
    pydantic_model.Sources.from_dict(config)
    
  # Verify the error message contains the expected text
  assert "Both bremsstrahlung_heat_sink and impurity_radiation_heat_sink with Mavrin model" in str(excinfo.value)
  assert "should not be active at the same time" in str(excinfo.value)
  
  # Test with only bremsstrahlung active (should pass validation)
  config = {
      'bremsstrahlung_heat_sink': {
          'mode': 'MODEL_BASED',
          'source_name': 'bremsstrahlung_heat_sink',
      },
      'impurity_radiation_heat_sink': {
          'model_function_name': 'impurity_radiation_mavrin_fit',
          'mode': 'ZERO',
          'source_name': 'impurity_radiation_heat_sink',
      },
      'j_bootstrap': {
          'mode': 'ZERO',
          'source_name': 'j_bootstrap',
      },
      'generic_current_source': {
          'mode': 'ZERO',
          'source_name': 'generic_current_source',
      },
      'qei_source': {
          'mode': 'ZERO',
          'source_name': 'qei_source',
      },
  }
  sources = pydantic_model.Sources.from_dict(config)
  
  # Test with only impurity radiation active (should pass validation)
  config = {
      'bremsstrahlung_heat_sink': {
          'mode': 'ZERO',
          'source_name': 'bremsstrahlung_heat_sink',
      },
      'impurity_radiation_heat_sink': {
          'model_function_name': 'impurity_radiation_mavrin_fit',
          'mode': 'MODEL_BASED',
          'source_name': 'impurity_radiation_heat_sink',
      },
      'j_bootstrap': {
          'mode': 'ZERO',
          'source_name': 'j_bootstrap',
      },
      'generic_current_source': {
          'mode': 'ZERO',
          'source_name': 'generic_current_source',
      },
      'qei_source': {
          'mode': 'ZERO',
          'source_name': 'qei_source',
      },
  }
  sources = pydantic_model.Sources.from_dict(config)