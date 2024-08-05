from pydantic import BaseModel, ConfigDict


class Field[T](BaseModel):
    """Represents an NXS Field (dataset), with attributes allowed"""

    value: T
    model_config = ConfigDict(arbitrary_types_allowed=True)


class NXobject(BaseModel):
    """
    The base NX object.
    """
