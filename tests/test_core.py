import numpy as np

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.constraints.spacing import spacing_violation
from wind_farm_opt.ml.features import extract_layout_features
from wind_farm_opt.optimization.layout import pack_layout, unpack_layout
from wind_farm_opt.physics.aep import AEPEvaluator
from wind_farm_opt.physics.power_curve import turbine_power_mw
from wind_farm_opt.physics.wake import jensen_wake_deficit


def test_power_curve_rated_region():
    assert turbine_power_mw(12.0) == 40.0
    assert turbine_power_mw(15.0) == 43.6


def test_wake_deficit_non_negative():
    deficit = jensen_wake_deficit(5.0, 5.0, 2.0, 5.0, 0.0, 8.0, 0.08)
    assert deficit >= 0.0


def test_spacing_violation_detects_overlap():
    x = np.array([0.0, 0.1])
    y = np.array([0.0, 0.0])
    assert spacing_violation(x, y, min_spacing=1.0) > 0.0


def test_pack_unpack_layout():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    packed = pack_layout(x, y)
    x2, y2 = unpack_layout(packed, 3)
    np.testing.assert_allclose(x, x2)
    np.testing.assert_allclose(y, y2)


def test_aep_evaluator_returns_positive_power():
    wind_map = np.full((10, 10), 8.0)
    config = SiteConfig(num_turbines=3)
    evaluator = AEPEvaluator(wind_map, config)
    x = np.array([2.0, 5.0, 7.0])
    y = np.array([2.0, 5.0, 7.0])
    metrics = evaluator.evaluate(x, y)
    assert metrics["aep_mw"] > 0.0


def test_feature_extraction_keys():
    wind_map = np.full((10, 10), 7.5)
    config = SiteConfig(num_turbines=3)
    features = extract_layout_features(
        wind_map,
        np.array([2.0, 5.0, 7.0]),
        np.array([2.0, 5.0, 7.0]),
        config,
    )
    assert "mean_wind_speed" in features
    assert features["mean_wind_speed"] == 7.5
