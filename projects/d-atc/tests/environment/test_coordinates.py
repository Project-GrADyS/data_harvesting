import numpy as np
import pytest
from pyproj import Geod

from d_atc.environment.coordinates import HomeCoordinateFrame


HOME = (-23.5505, -46.6333, 760.0)


def _longitude_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return (actual - expected + 180.0) % 360.0 - 180.0


def test_home_is_local_origin_and_round_trips() -> None:
    frame = HomeCoordinateFrame(HOME)

    local = frame.geographic_to_local(*HOME)
    geographic = frame.local_to_geographic(0.0, 0.0, 0.0)

    assert local == pytest.approx((0.0, 0.0, 0.0), abs=1e-8)
    assert geographic == pytest.approx(HOME, abs=1e-10)


def test_cardinal_directions_have_expected_axes() -> None:
    frame = HomeCoordinateFrame(HOME)
    geod = Geod(ellps="WGS84")

    east_lon, east_lat, _ = geod.fwd(HOME[1], HOME[0], 90.0, 1_000.0)
    west_lon, west_lat, _ = geod.fwd(HOME[1], HOME[0], 270.0, 1_000.0)
    north_lon, north_lat, _ = geod.fwd(HOME[1], HOME[0], 0.0, 1_000.0)
    south_lon, south_lat, _ = geod.fwd(HOME[1], HOME[0], 180.0, 1_000.0)

    east = frame.geographic_to_local(east_lat, east_lon, HOME[2])
    west = frame.geographic_to_local(west_lat, west_lon, HOME[2])
    north = frame.geographic_to_local(north_lat, north_lon, HOME[2])
    south = frame.geographic_to_local(south_lat, south_lon, HOME[2])

    assert east == pytest.approx((1_000.0, 0.0, 0.0), abs=1e-5)
    assert west == pytest.approx((-1_000.0, 0.0, 0.0), abs=1e-5)
    assert north == pytest.approx((0.0, 1_000.0, 0.0), abs=1e-5)
    assert south == pytest.approx((0.0, -1_000.0, 0.0), abs=1e-5)


def test_altitude_only_changes_up_axis() -> None:
    frame = HomeCoordinateFrame(HOME)

    assert frame.geographic_to_local(HOME[0], HOME[1], HOME[2] + 250.0) == pytest.approx(
        (0.0, 0.0, 250.0),
        abs=1e-8,
    )


@pytest.mark.parametrize("count", [100, 1_000, 5_000])
def test_batched_coordinates_round_trip_at_bluesky_scales(count: int) -> None:
    frame = HomeCoordinateFrame(HOME)
    rng = np.random.default_rng(42)
    bearings = rng.uniform(0.0, 360.0, count)
    distances = rng.uniform(0.0, 100_000.0, count)
    longitudes, latitudes, _ = Geod(ellps="WGS84").fwd(
        np.full(count, HOME[1]),
        np.full(count, HOME[0]),
        bearings,
        distances,
    )
    altitudes = rng.uniform(0.0, 12_000.0, count)

    east, north, up = frame.geographic_to_local(latitudes, longitudes, altitudes)
    actual_latitudes, actual_longitudes, actual_altitudes = frame.local_to_geographic(
        east,
        north,
        up,
    )

    assert isinstance(east, np.ndarray)
    np.testing.assert_allclose(actual_latitudes, latitudes, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        _longitude_error(np.asarray(actual_longitudes), longitudes),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(actual_altitudes, altitudes, atol=1e-10, rtol=0.0)


def test_antimeridian_round_trip() -> None:
    frame = HomeCoordinateFrame((10.0, 179.9, 100.0))
    latitudes = np.array([10.0, 10.1])
    longitudes = np.array([-179.9, 179.8])
    altitudes = np.array([100.0, 500.0])

    local = frame.geographic_to_local(latitudes, longitudes, altitudes)
    actual_latitudes, actual_longitudes, actual_altitudes = frame.local_to_geographic(*local)

    np.testing.assert_allclose(actual_latitudes, latitudes, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(
        _longitude_error(np.asarray(actual_longitudes), longitudes),
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(actual_altitudes, altitudes, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("home", "message"),
    [
        ((float("nan"), 0.0, 0.0), "finite"),
        ((91.0, 0.0, 0.0), "latitude"),
        ((0.0, 181.0, 0.0), "longitude"),
    ],
)
def test_invalid_home_is_rejected(home: tuple[float, float, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HomeCoordinateFrame(home)


def test_invalid_and_incompatible_coordinate_inputs_are_rejected() -> None:
    frame = HomeCoordinateFrame(HOME)

    with pytest.raises(ValueError, match="latitude"):
        frame.geographic_to_local(91.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        frame.local_to_geographic(float("inf"), 0.0, 0.0)
    with pytest.raises(ValueError, match="broadcast-compatible"):
        frame.geographic_to_local([0.0, 1.0], [0.0, 1.0, 2.0], 0.0)
