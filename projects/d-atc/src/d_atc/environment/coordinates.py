from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer
from pyproj.enums import TransformDirection


CoordinateInput: TypeAlias = float | npt.ArrayLike
CoordinateOutput: TypeAlias = float | npt.NDArray[np.float64]


def _broadcast_coordinates(
    first: CoordinateInput,
    second: CoordinateInput,
    third: CoordinateInput,
    *,
    names: tuple[str, str, str],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
    try:
        arrays = np.broadcast_arrays(
            np.asarray(first, dtype=np.float64),
            np.asarray(second, dtype=np.float64),
            np.asarray(third, dtype=np.float64),
        )
    except ValueError as error:
        raise ValueError(f"{', '.join(names)} must have broadcast-compatible shapes") from error

    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{', '.join(names)} must contain only finite values")

    return arrays[0], arrays[1], arrays[2], arrays[0].ndim == 0


def _coordinate_output(
    value: CoordinateInput,
    *,
    scalar: bool,
) -> CoordinateOutput:
    array = np.asarray(value, dtype=np.float64)
    return float(array) if scalar else array


class HomeCoordinateFrame:
    """Convert WGS-84 coordinates to a home-centered local aviation frame.

    The local axes are east, north, and altitude above home, all in meters.
    Horizontal coordinates use an azimuthal-equidistant projection centered
    at ``home``.
    """

    def __init__(self, home: tuple[float, float, float]) -> None:
        latitude, longitude, altitude = home
        home_values = np.asarray((latitude, longitude, altitude), dtype=np.float64)
        if not np.all(np.isfinite(home_values)):
            raise ValueError("home must contain only finite values")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("home latitude must be in the range [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("home longitude must be in the range [-180, 180]")

        self.home = (float(latitude), float(longitude), float(altitude))
        local_crs = CRS.from_proj4(
            "+proj=aeqd "
            f"+lat_0={latitude!r} +lon_0={longitude!r} "
            "+datum=WGS84 +units=m +no_defs +type=crs"
        )
        self._transformer = Transformer.from_crs(
            "EPSG:4326",
            local_crs,
            always_xy=True,
        )

    def geographic_to_local(
        self,
        latitude: CoordinateInput,
        longitude: CoordinateInput,
        altitude: CoordinateInput,
    ) -> tuple[CoordinateOutput, CoordinateOutput, CoordinateOutput]:
        """Convert latitude/longitude/altitude to east/north/up."""
        latitudes, longitudes, altitudes, scalar = _broadcast_coordinates(
            latitude,
            longitude,
            altitude,
            names=("latitude", "longitude", "altitude"),
        )
        if np.any((latitudes < -90.0) | (latitudes > 90.0)):
            raise ValueError("latitude must be in the range [-90, 90]")

        east, north = self._transformer.transform(longitudes, latitudes, errcheck=True)
        up = altitudes - self.home[2]
        return (
            _coordinate_output(east, scalar=scalar),
            _coordinate_output(north, scalar=scalar),
            _coordinate_output(up, scalar=scalar),
        )

    def local_to_geographic(
        self,
        east: CoordinateInput,
        north: CoordinateInput,
        up: CoordinateInput,
    ) -> tuple[CoordinateOutput, CoordinateOutput, CoordinateOutput]:
        """Convert east/north/up to latitude/longitude/altitude."""
        eastings, northings, up_values, scalar = _broadcast_coordinates(
            east,
            north,
            up,
            names=("east", "north", "up"),
        )
        longitude, latitude = self._transformer.transform(
            eastings,
            northings,
            direction=TransformDirection.INVERSE,
            errcheck=True,
        )
        altitude = up_values + self.home[2]
        return (
            _coordinate_output(latitude, scalar=scalar),
            _coordinate_output(longitude, scalar=scalar),
            _coordinate_output(altitude, scalar=scalar),
        )
