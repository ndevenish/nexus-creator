import itertools
import sys
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import glob
import textwrap
from xsdata_pydantic.bindings import XmlParser
from pydantic import BaseModel, ConfigDict

from . import nxdl
from .nxdl import Definition, DocType
from typing import Any, Self, TypedDict, Unpack


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


class Field[T](BaseModel):
    """Represents an NXS Field (dataset), with attributes allowed"""

    value: T
    model_config = ConfigDict(arbitrary_types_allowed=True)


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
}

IMPORTS = """
import datetime
from pint import Quantity
from typing import Literal
from .core import NXobject, Field
"""


def _resolve_type(type_value: list | Any, optional: bool = True) -> str:
    if isinstance(type_value, list):
        assert len(type_value) == 1
        type_value = type_value[0]
    if optional:
        return type_maps[type_value] + " | None = None"
    else:
        return type_maps[type_value]


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
            type=_resolve_type(attr.type_value, optional=attr.optional),
            doc=_convert_doc(attr.doc),
        )


class ClassDefinition(BaseModel):
    name: str
    parent: str
    doc: str | None = None
    attributes: list[ClassAttribute] = []
    fields: list[ClassAttribute] = []
    groups: list[ClassAttribute] = []

    class ClassDefinitionKwargs(TypedDict, total=False):
        """Exists to duplicate the class init to allow passing in doc for coercion"""

        name: str
        parent: str
        doc: str | DocType | list[DocType] | None
        attributes: list[ClassAttribute]
        fields: list[ClassAttribute]
        groups: list[ClassAttribute]

    def __init__(self, **kwargs: Unpack[ClassDefinitionKwargs]):
        super().__init__(**kwargs)

    def __str__(self) -> str:
        parts = [f"class {self.name}({self.parent}):"]
        if self.doc:
            parts.append(
                textwrap.indent(
                    '"""\n' + _prepare_paragraphs(self.doc) + '\n"""', "    "
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
        # if k == "attribute":
        #     breakpoint()
        if getattr(field, k) != v.get_default(call_default_factory=True):
            to_set[k] = getattr(field, k)
    # Things we implicitly have access to
    to_set.pop("doc", None)
    to_set.pop("type_value", None)
    to_set.pop("enumeration", None)
    to_set.pop("attribute", None)

    # name is required but redundant for our purposes. Don't give a
    # result if it's the only value
    if to_set.keys() == {"name"}:
        return None
    return "FieldType(" + ", ".join(f"{k}={v!r}" for k, v in to_set.items()) + ")"


def _create_attribute_subclass(
    attributes: list[nxdl.AttributeType], field_name: str, definition_name: str
) -> ClassDefinition:
    # Work out what to call this
    name = f"Field {field_name}".replace("_", " ").title().replace(" ", "")
    # If we only have one attribute, name for that
    if len(attributes) == 1:
        name = "Field" + attributes[0].name.replace("_", " ").title().replace(" ", "")

    return ClassDefinition(
        name=name,
        parent="Field",
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
        # Holds separate line parts for the class body output
        assert isinstance(defn.extends, str)
        new_class = ClassDefinition(name=defn.name, parent=defn.extends, doc=defn.doc)

        # Now, handle attributes
        for attr in defn.attribute:
            # Attributes are simple values, stored on the group (or
            # dataset) itself. These can just get added as plain
            # properties onto the output class.
            new_class.attributes.append(ClassAttribute.from_attribute(attr))

            # Unhandled things that we might want to do later
            assert not attr.enumeration
            assert not attr.dimensions

        for group in defn.group:
            name = group.name or group.type_value[2:]
            group_type = f"list[{group.type_value}]"
            if group.optional or group.min_occurs == 0:
                # If optional, then default to an empty list
                group_type += " = []"

            new_class.groups.append(
                ClassAttribute(name=name, type=group_type, doc=_convert_doc(group.doc))
            )

            assert not group.group, "Groups (typed?) contains groups?!?!?"
            assert group.max_occurs is None
            assert group.min_occurs is None or group.min_occurs == 0
            # Things we might be able to handle, once we see instances of
            assert not group.attribute, "Groups can have defined attributes, redundantly with their type definition? How to handle?"
            if group.field_value:
                print(
                    f"Warning: Found field definition on object {defn.name}.{name}. This is redundant? Check this is truly redundant later"
                )
            assert not group.choice
            assert not group.link

        # Now, process the fields. Fields are complex; they are datasets, but can
        # contain extra attribute information, so we need to use a wrapped value
        # object to represent them here.
        for field in defn.field_value:
            optional = field.optional or field.min_occurs == 0

            # Work out what data type we need to use for this
            base_type = _resolve_type(field.type_value, optional=False)
            if field.enumeration:
                field_type = f"Literal[{', '.join(repr(x.value) for x in field.enumeration.item)}]"
            else:
                if field.units:
                    # We have a dimensioned unit
                    # Note that this is currently invalid pint declaration, we
                    # need to find a way to make this declarable (dimension or plain unit)
                    field_type = f"Quantity[{base_type}]"
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
            if optional:
                field_type += " | None"

            # If we have a complex (or non-default) field spec, annotate the type with it
            if field_annotation := _field_repr(field):
                field_type = f"Annotated[{field_type}, {field_annotation}]"

            if optional:
                field_type += " = None"

            new_class.fields.append(
                ClassAttribute(
                    name=field.name, type=field_type, doc=_convert_doc(field.doc)
                )
            )

            # Other things we can't or don't yet handle
            assert field.name_type == nxdl.FieldTypeNameType.SPECIFIED
            assert not field.dimensions
            assert not field.dimensions and field.max_occurs == 1
            assert field.min_occurs == 0

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

    print("\n\n".join(str(x) for x in generate_classes))
