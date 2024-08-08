from functools import partial
from typing import Any
from pydantic import AfterValidator, BaseModel, ConfigDict, TypeAdapter
from pint import Quantity


def _check_dimension(dim: str, value: Quantity) -> Quantity:
    """Checks that a Quantity value has a specific dimensionality"""
    if not value.check(dim):
        raise ValueError(f"Quantity dimensions do not match '{dim}'")
    return value


def _check_quantity_type(is_type: type, value: Quantity) -> Quantity:
    print(f"Checking {value=} matches type {is_type}")
    TypeAdapter(is_type).validate_python(value.m)
    return value


class Units:
    """Validators for validating the dimensionality of a given quantity"""

    # "NX_ANY": None,
    # # NX_TRANSFORMATION is special, because it changes depending on the type of transform
    # "NX_TRANSFORMATION": None,

    ANGLE = AfterValidator(partial(_check_dimension, "[]"))
    AREA = AfterValidator(partial(_check_dimension, "[area]"))
    CROSS_SECTION = AfterValidator(partial(_check_dimension, "[area]"))
    CHARGE = AfterValidator(partial(_check_dimension, "[charge]"))
    CURRENT = AfterValidator(partial(_check_dimension, "[current]"))
    DIMENSIONLESS = AfterValidator(partial(_check_dimension, "[]"))
    EMITTANCE = AfterValidator(partial(_check_dimension, "[length] * [area]"))
    ENERGY = AfterValidator(partial(_check_dimension, "[energy]"))
    FLUX = AfterValidator(partial(_check_dimension, "1 / [time] / [area]"))
    FREQUENCY = AfterValidator(partial(_check_dimension, "[frequency]"))
    LENGTH = AfterValidator(partial(_check_dimension, "[length]"))
    MASS = AfterValidator(partial(_check_dimension, "[mass]"))
    MASS_DENSITY = AfterValidator(partial(_check_dimension, "[density]"))
    MOLECULAR_WEIGHT = AfterValidator(partial(_check_dimension, "[mass] / [substance]"))
    PER_AREA = AfterValidator(partial(_check_dimension, "1 / [area]"))
    PER_LENGTH = AfterValidator(partial(_check_dimension, "1 / [length]"))
    PERIOD = AfterValidator(partial(_check_dimension, "[time]"))
    POWER = AfterValidator(partial(_check_dimension, "[power]"))
    PRESSURE = AfterValidator(partial(_check_dimension, "[pressure]"))
    PULSES = AfterValidator(partial(_check_dimension, "[]"))
    COUNT = AfterValidator(partial(_check_dimension, "[]"))
    SCATTERING_LENGTH_DENSITY = AfterValidator(partial(_check_dimension, "[area]"))
    SOLID_ANGLE = AfterValidator(partial(_check_dimension, "[]"))
    TEMPERATURE = AfterValidator(partial(_check_dimension, "[temperature]"))
    TIME = AfterValidator(partial(_check_dimension, "[time]"))
    TIME_OF_FLIGHT = AfterValidator(partial(_check_dimension, "[time]"))
    UNITLESS = AfterValidator(partial(_check_dimension, "[]"))
    VOLTAGE = AfterValidator(partial(_check_dimension, "[electric_potential]"))
    VOLUME = AfterValidator(partial(_check_dimension, "[volume]"))
    WAVELENGTH = AfterValidator(partial(_check_dimension, "[length]"))
    WAVENUMBER = AfterValidator(partial(_check_dimension, "[]"))


def QuantityType[T](is_type: type) -> Any:
    """Create a validator that a pint Quantity contains a particular type"""
    return AfterValidator(partial(_check_quantity_type, is_type))


class Field[T](BaseModel):
    """Represents an NXS Field (dataset), with attributes allowed"""

    value: T
    model_config = ConfigDict(arbitrary_types_allowed=True)


class NXobject(BaseModel):
    """
    The base NX object.
    """

    model_config = ConfigDict(validate_assignment=True)


__all__ = [
    "Units",
    "QuantityType",
    "Field",
    "NXobject",
]
