"""Test script to verify issue #1744 fix.

This script tests that IMAS geometry can be loaded from equilibrium_object
and the config can be successfully serialized to JSON.
"""
import copy
import json

from torax._src import path_utils
from torax._src.imas_tools.input import loader
from torax.tests.test_data import test_iterhybrid_predictor_corrector
from torax._src.torax_pydantic import model_config

# Load IDSs
path = (
    path_utils.torax_path()
    / "data"
    / "imas_data"
    / "ITERhybrid_COCOS17_IDS_ddv4.nc"
)
equilibrium_ids = loader.load_imas_data(str(path), "equilibrium")

CONFIG = copy.deepcopy(test_iterhybrid_predictor_corrector.CONFIG)

CONFIG["geometry"] = {
    "geometry_type": "imas",
    "equilibrium_object": equilibrium_ids,
    "imas_filepath": None,
    "Ip_from_parameters": True,
}

print("Creating ToraxConfig with equilibrium_object...")
torax_config = model_config.ToraxConfig.from_dict(CONFIG)

print("Testing model_dump()...")
config_dict = torax_config.model_dump()
print("✓ model_dump() succeeded")

print("\nTesting model_dump_json()...")
try:
    config_json = torax_config.model_dump_json()
    print("✓ model_dump_json() succeeded")

    # Verify the JSON is valid
    parsed = json.loads(config_json)
    print("✓ JSON is valid and parseable")

    # Verify equilibrium_object is None in serialized output
    assert parsed['geometry']['equilibrium_object'] is None, \
        "equilibrium_object should be None in serialized output"
    print("✓ equilibrium_object is correctly excluded (set to None)")

    print("\n✅ All tests passed! Issue #1744 is fixed.")

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")
    raise
