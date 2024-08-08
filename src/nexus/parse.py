from __future__ import annotations

import itertools
import sys
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import glob
import textwrap
from xsdata_pydantic.bindings import XmlParser
from pydantic import BaseModel

from . import nxdl
from . import core
from .nxdl import Definition, DocType
from typing import Any, Self


def _convert_doc(v: DocType | list[DocType] | None) -> str | None:
    """Take a DocType (or list of), and convert to a single, plain string"""
    if v is None:
        return None
    if isinstance(v, DocType):
        v = [v]
    out: str = "\n\n".join(
        itertools.chain.from_iterable(
            [textwrap.dedent(str(y)) for y in x.content] for x in v
        )
    )
    return "\n".join(x.rstrip() for x in out.splitlines())


class NXobject(BaseModel):
    """
    The base NX object.
    """


# Mapping of which type annotations to give each NX type
type_maps = {
    "NX_CHAR": "str",
    "NX_INT": "int",
    "NX_BOOLEAN": "bool",
    "NX_FLOAT": "float",
    "NX_NUMBER": "int | float",
    "NX_DATE_TIME": "datetime.datetime",
    "NX_CHAR_OR_NUMBER": "str | int | float",
    "NX_BINARY": "bytes",
    "NX_POSINT": "PositiveInt",
    # "ISO8601": ,
    # "NX_CCOMPLEX": ,
    # "NX_COMPLEX": ,
    # "NX_PCOMPLEX": ,
    # "NX_QUATERNION": ,
    # "NX_UINT": ,
}
# Mapping the pint dimensionality name from the nexus unit name
dimensions_map = {
    # NX_ANY we just ignore
    # "NX_ANY": None,
    # NX_TRANSFORMATION is special, because it changes depending on the type of transform
    # "NX_TRANSFORMATION": None,
    "NX_ANGLE": "[]",
    "NX_AREA": "[area]",
    "NX_CROSS_SECTION": "[area]",
    "NX_CHARGE": "[charge]",
    "NX_CURRENT": "[current]",
    "NX_DIMENSIONLESS": "[]",
    "NX_EMITTANCE": "[length] * [area]",
    "NX_ENERGY": "[energy]",
    "NX_FLUX": "1 / [time] / [area]",
    "NX_FREQUENCY": "[frequency]",
    "NX_LENGTH": "[length]",
    "NX_MASS": "[mass]",
    "NX_MASS_DENSITY": "[density]",
    "NX_MOLECULAR_WEIGHT": "[mass] / [substance]",
    "NX_PER_AREA": "1 / [area]",
    "NX_PER_LENGTH": "1 / [length]",
    "NX_PERIOD": "[time]",
    "NX_POWER ": "[power]",
    "NX_PRESSURE": "[pressure]",
    "NX_PULSES": "[]",
    "NX_COUNT": "[]",
    "NX_SCATTERING_LENGTH_DENSITY": "[area]",
    "NX_SOLID_ANGLE": "[]",
    "NX_TEMPERATURE": "[temperature]",
    "NX_TIME": "[time]",
    "NX_TIME_OF_FLIGHT": "[time]",
    "NX_UNITLESS": "[]",
    "NX_VOLTAGE": "[electric_potential]",
    "NX_VOLUME": "[volume]",
    "NX_WAVELENGTH": "[length]",
    "NX_WAVENUMBER": "[]",
}

GENERATED_HEADER = """
from __future__ import annotations

import datetime
from typing import Annotated, Literal

from annotated_types import MinLen
from h5py import Dataset, ExternalLink
from pint import Quantity
from pydantic import PositiveInt

from .core import Field, NXobject, QuantityType, Units
"""


def _resolve_type(
    type_value: list | Any,
    optional: bool = True,
    enumeration: nxdl.EnumerationType | None = None,
) -> str:
    if isinstance(type_value, list):
        # Sometimes type value comes out of parsing the definitions as a list
        assert len(type_value) == 1
        type_value = type_value[0]
    if enumeration:
        type_value = f"Literal[{', '.join(repr(x.value) for x in enumeration.item)}]"
    else:
        type_value = type_maps[type_value]

    return type_value + (" | None = None" if optional else "")


def _prepare_paragraphs(
    data: str, *, width: int = 70, indent: str | None = None
) -> str:
    """Wrap and optionally indent text lines, but split between paragraphs"""
    if indent:
        # Account for indent when calculating wrapping point
        width -= len(indent)
    output = "\n\n".join(
        "\n".join(textwrap.wrap(p.strip(), width=width, break_long_words=False))
        for p in data.split("\n\n")
    )
    if indent:
        if indent.isspace():
            output = textwrap.indent(output, indent)
        else:
            # If we've been given non-whitespace indent
            output = textwrap.indent(output, indent, lambda _: True)

    return output


class ClassAttribute(BaseModel):
    name: str
    type: str
    doc: str | None

    def __str__(self) -> str:
        if self.doc:
            doccomment = _prepare_paragraphs(self.doc, indent="# ")
            return f"{doccomment}\n{self.name} : {self.type}"
        else:
            return f"{self.name} : {self.type}"

    @classmethod
    def from_attribute(cls, attr: nxdl.AttributeType) -> Self:
        return cls(
            name=attr.name,
            type=_resolve_type(
                attr.type_value, optional=attr.optional, enumeration=attr.enumeration
            ),
            doc=_convert_doc(attr.doc),
        )


class ClassDefinition(BaseModel):
    name: str
    generics: list[str] = []
    parent: str
    doc: str | None = None
    attributes: list[ClassAttribute] = []
    fields: list[ClassAttribute] = []
    groups: list[ClassAttribute] = []

    def __contains__(self, name: str) -> bool:
        return any(
            name == x.name
            for x in itertools.chain(self.attributes, self.fields, self.groups)
        )

    def __str__(self) -> str:
        generics = ""
        if self.generics:
            generics = f"[{', '.join(self.generics)}]"

        parts = [f"class {self.name}{generics}({self.parent}):"]
        if self.doc:
            str_type = "r" if "\\" in self.doc else ""
            parts.append(
                textwrap.indent(
                    f'{str_type}"""\n' + _prepare_paragraphs(self.doc) + '\n"""', "    "
                ),
            )

        blocks = []
        for block in [self.attributes, self.groups, self.fields]:
            if block:
                blocks.append(
                    "\n".join(textwrap.indent(str(x), "    ") + "\n" for x in block)
                )

        if not blocks and not self.doc:
            parts.append("    pass")
        else:
            parts.append("\n\n".join(blocks))

        return "\n".join(parts)


def _field_repr(field: nxdl.FieldType) -> str | None:
    """
    Construct a minimized FieldType declaration, removing things we already know
    """
    to_set = {}
    for k, v in field.model_fields.items():
        if getattr(field, k) != v.get_default(call_default_factory=True):
            to_set[k] = getattr(field, k)
    # Things we implicitly have access to
    to_set.pop("doc", None)
    to_set.pop("type_value", None)
    to_set.pop("enumeration", None)
    to_set.pop("attribute", None)
    to_set.pop("units", None)
    to_set.pop("name_type", None)
    to_set.pop("min_occurs", None)
    to_set.pop("max_occurs", None)

    # name is required but redundant for our purposes. Don't give a
    # result if it's the only value
    if to_set.keys() == {"name"}:
        return None
    return "FieldType(" + ", ".join(f"{k}={v!r}" for k, v in to_set.items()) + ")"


def _create_attribute_subclass(
    attributes: list[nxdl.AttributeType], field_name: str, _definition_name: str
) -> ClassDefinition:
    """Create a new subclass to hold declared field attributes."""
    # Work out what to call this
    name = f"Field {field_name}".replace("_", " ").title().replace(" ", "")
    # If we only have one attribute, name for that
    if len(attributes) == 1:
        name = "Field" + attributes[0].name.replace("_", " ").title().replace(" ", "")

    return ClassDefinition(
        name=name,
        generics=["T"],
        parent="Field[T]",
        attributes=[ClassAttribute.from_attribute(attr) for attr in attributes],
    )


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "sources",
        nargs="*",
        default=["v2024.2/base_classes/NXsource.nxdl.xml"],
    )
    args = parser.parse_args()

    # Expand every passed file as if it contained a glob
    args.sources = list(
        chain(*[[Path(x) for x in glob.glob(p, recursive=True)] for p in args.sources])
    )
    if not args.sources:
        sys.exit("Error: No sources found after expansion")

    definitions: list[Definition] = []
    for source in args.sources:
        parser = XmlParser()
        definitions.append(parser.parse(source, Definition))

    # Don't regenerate NXobject
    definitions = [x for x in definitions if not x.name == "NXobject"]
    generate_classes: list[ClassDefinition] = []

    for defn in definitions:
        print(f"Processing {defn.name}", file=sys.stderr)
        # Keep track of the class definition parts as we go
        assert isinstance(defn.extends, str)
        new_class = ClassDefinition(
            name=defn.name, parent=defn.extends, doc=_convert_doc(defn.doc)
        )

        # First, handle attributes. These are simple values, stored on
        # the group (or dataset) itself. These can just get added as
        # plain properties onto the output class.
        for attr in defn.attribute:
            assert attr.name not in new_class
            new_class.attributes.append(ClassAttribute.from_attribute(attr))

            # Unhandled things that we might want to do later
            assert not attr.dimensions

        # Now, handle groups. These are other nexus classes represented as HDF5
        # group objects, and we can have multiple instances of each.
        for group in defn.group:
            group_annotations = []
            name = group.name or group.type_value[2:]
            group_type = f"list[{group.type_value}]"
            if group.optional or group.min_occurs == 0:
                # If optional, then default to an empty list
                group_type += " = []"

            assert (
                not group.group
            ), f"Group {defn.name}{group.name} contains other groups groups?!?!?"
            assert group.max_occurs is None

            if group.min_occurs:
                # MinLen (from annotated-types) is easier to read than
                # pydantic.Field, especially when we are trying not to
                # conflict names.
                group_annotations.append(f"MinLen({group.min_occurs})")

            if group.attribute:
                # This.. happens, and adds extra definitions onto other
                # objects. Not sure how to handle yet.
                print(
                    f"Warning: Found group ({defn.name}.{group.name}) with declared attributes ({', '.join(x.name for x in group.attribute)}), is this redundant?",
                    file=sys.stderr,
                )
            if group.field_value:
                print(
                    f"Warning: Found field definition on object {defn.name}.{name}. This is redundant? Check this is truly redundant later",
                    file=sys.stderr,
                )

            # Other things that we have not seen happen, technically in spec?
            assert not group.choice
            assert not group.link

            if group_annotations:
                group_type = f"Annotated[{group_type}, {', '.join(group_annotations)}]"

            assert (
                name not in new_class
            ), f"Group {name} already exists, we are trying to generate twice?"
            new_class.groups.append(
                ClassAttribute(name=name, type=group_type, doc=_convert_doc(group.doc))
            )

        # Now, process the fields. Fields are complex; they are datasets, but can
        # contain extra attribute information, so we need to use a wrapped value
        # object to represent them here.
        for field in sorted(defn.field_value, key=lambda x: x.name):
            optional = field.optional or field.min_occurs == 0
            # Collate any annotations to apply
            field_annotations = []

            # Work out what data type we need to use for this
            base_type = _resolve_type(
                field.type_value, optional=False, enumeration=field.enumeration
            )
            # A couple of fields don't have units, but specify that units is present in
            # the attribute list. Just treat this as a Quantity.
            if unit_attr := [x for x in field.attribute if x.name == "units"]:
                field.attribute.remove(unit_attr[0])
                field.units = "NX_ANY"
                print(
                    f"Warning: Field {defn.name}.{field.name} has 'units' attribute. Should this just be a quantity?",
                    file=sys.stderr,
                )
            if field.units:
                # We have a dimensioned unit
                field_type = "Quantity"
                field_annotations.append(f"QuantityType({base_type})")
                units = field.units.removeprefix("NX_")
                if units == "ANY":
                    pass
                elif units == "TRANSFORMATION":
                    print(
                        f"Warning: Encountered NX_TRANSFORMATION on class {defn.name}. This is currently unhandled by typing.",
                        file=sys.stderr,
                    )
                else:
                    assert hasattr(core.Units, units), f"Unknown unit: {units}"
                    field_annotations.append(f"Units.{field.units.removeprefix("NX_")}")

            else:
                field_type = base_type

            # Fields can have lists of declared attributes. We handle this by
            # having explicit Field subclasses with named attributes containing
            # the meta-information about what type they hold.
            #
            # If no specific declared attributes, we still use a Field instance
            # both for consistency and for the ability to set more.
            field_superclass = "Field"
            if field.attribute:
                attr_class = _create_attribute_subclass(
                    field.attribute, field.name, defn.name
                )
                generate_classes.append(attr_class)
                field_superclass = attr_class.name

            field_type = f"{field_superclass}[{field_type}]"

            # If we have "ANY" naming, then this field occur multiple times,
            # and
            if field.name_type is nxdl.FieldTypeNameType.ANY:
                field_type = f"dict[str, {field_type}]"

            # If we have a complex (or non-default) field spec, annotate the type with it
            if source_field_decl := _field_repr(field):
                field_annotations.append(source_field_decl)

            if field_annotations:
                field_type = f"Annotated[{field_type}, {', '.join(field_annotations)}]"

            field_type += " | ExternalLink"
            if optional:
                field_type += " | None"
            if optional:
                field_type += " = None"

            assert field.name not in new_class.attributes
            new_class.fields.append(
                ClassAttribute(
                    name=field.name, type=field_type, doc=_convert_doc(field.doc)
                )
            )

            # Other things we can't or don't yet handle
            # assert not field.dimensions
            assert not (field.units and field.enumeration)
            if field.max_occurs == nxdl.NonNegativeUnboundedValue.UNBOUNDED:
                print(
                    f"Warning: Field {defn.name}.{field.name} has unbounded multiplicity; how does this happen on a field? Ignoring.",
                    file=sys.stderr,
                )
            else:
                assert (
                    (field.min_occurs == 0 and field.max_occurs in {0, 1})
                    or (field.min_occurs == 1 and field.max_occurs == 1)
                ), f"It's expected min/max_occurs only used to mark optionality ({field})"

            # Other things not handled, that otherwise may be present on
            # the field annotation:
            # - units
            # - long_name
            # - signal
            # - primary
            # - stride
            # - data_offset
            # - interpretation

        generate_classes.append(new_class)

    print(GENERATED_HEADER)
    print("\n\n".join(str(x) for x in generate_classes))
