from functools import partial
from typing import Any
from pydantic import AfterValidator, BaseModel, ConfigDict, TypeAdapter
from pint import Quantity


def _check_dimension(dim: str, value: Quantity) -> Quantity:
    """Checks that a Quantity value has a specific dimensionality"""
    assert value.check(dim), f"Quantity dimensions do not match '{dim}'"
    return value


def _check_quantity_type(is_type: type, value: Quantity) -> Quantity:
    print(f"Checking {value=} matches type {is_type}")
    TypeAdapter(is_type).validate_python(value.m)
    return value


class NXUnit:
    """Validators for validating the dimensionality of a given quantity"""

    LENGTH = AfterValidator(partial(_check_dimension, "[length]"))


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


__all__ = [
    "NXUnit",
    "QuantityType",
    "Field",
    "NXobject",
]
