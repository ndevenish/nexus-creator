import itertools
import sys
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import glob
import textwrap
from xsdata_pydantic.bindings import XmlParser
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import BeforeValidator

from . import nxdl
from .nxdl import Definition, DocType
from typing import Any, Annotated


def _convert_doc(doc: DocType | list[DocType]) -> str:
    """Take a DocType (or list of), and convert to a single, plain string"""
    if isinstance(doc, DocType):
        doc = [doc]
    full_text: str = "\n\n".join(
        itertools.chain.from_iterable(
            [textwrap.dedent(y) for y in x.content] for x in doc
        )
    )
    return "\n".join(x.rstrip() for x in full_text.splitlines())


# Something to convert nxdl DocTypes to String
def _convert_doc_validator(v: Any) -> Any:
    if isinstance(v, DocType) or isinstance(v, list):
        return _convert_doc(v)
    return v


type ParsedDoc = Annotated[str, BeforeValidator(_convert_doc_validator)]


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

    print(f"{output=}")
    return output


class ClassAttribute(BaseModel):
    name: str
    type: str
    doc: ParsedDoc | None

    def __str__(self) -> str:
        if self.doc:
            doccomment = _prepare_paragraphs(self.doc, indent="# ")
            return f"{doccomment}\n{self.name} : {self.type}"
        else:
            return f"{self.name} : {self.type}"


class ClassDefinition(BaseModel):
    name: str
    parent: str
    doc: ParsedDoc | None = None
    attributes: list[ClassAttribute] = []
    fields: list[ClassAttribute] = []
    groups: list[ClassAttribute] = []

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

    output = [IMPORTS]

    for defn in definitions:
        # Holds separate line parts for the class body output
        new_class = ClassDefinition(name=defn.name, parent=defn.extends, doc=defn.doc)

        # Now, handle attributes
        for attr in defn.attribute:
            # Attributes are simple values, stored on the group (or
            # dataset) itself. These can just get added as plain
            # properties onto the output class.
            attr_type = _resolve_type(attr.type_value, optional=attr.optional)
            new_class.attributes.append(
                ClassAttribute(name=attr.name, type=attr_type, doc=attr.doc)
            )

            # Unhandled things that we might want to do later
            assert not attr.enumeration
            assert not attr.dimensions

        for group in defn.group:
            name = group.name or group.type_value[2:]
            new_class.groups.append(
                ClassAttribute(
                    name=name, type=f"list[{group.type_value}]", doc=group.doc
                )
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
                # breakpoint()
                if optional:
                    field_type += " | None = None"
            else:
                if field.units:
                    # We have a dimensioned unit
                    field_type = f"Quantity[{base_type}]"
                else:
                    field_type = base_type
                if optional:
                    field_type += " | None = None"

            new_class.fields.append(
                ClassAttribute(name=field.name, type=field_type, doc=field.doc)
            )

            # Other things we can't or don't handle
            assert field.name_type == nxdl.FieldTypeNameType.SPECIFIED

        print(str(new_class))

        # dimensions: dimensions of a data element in a NeXus file
        # attribute: attributes to be used with this field
        # enumeration: A field can specify which values are to be used
        # units: String describing the engineering units. The string
        #     should be appropriate for the value and should conform to
        #     the NeXus rules for units. Conformance is not validated at
        #     this time.
        # long_name: Descriptive name for this field (may include
        #     whitespace and engineering units). Often, the long_name
        #     (when defined) will be used as the axis label on a plot.
        # signal: Presence of the ``signal`` attribute means this field is
        #     an ordinate. Integer marking this field as plottable data
        #     (ordinates). The value indicates the priority of selection
        #     or interest. Some facilities only use ``signal=1`` while
        #     others use ``signal=2`` to indicate plottable data of
        #     secondary interest. Higher numbers are possible but not
        #     common and interpretation is not standard. A field with a
        #     ``signal`` attribute should not have an ``axis`` attribute.
        # primary: Integer indicating the priority of selection of this
        #     field for plotting (or visualization) as an axis. Presence
        #     of the ``primary`` attribute means this field is an
        #     abscissa.
        # type_value: Defines the type of the element as allowed by NeXus.
        #     See :ref:`here&lt;Design-DataTypes&gt;` and
        #     :ref:`elsewhere&lt;nxdl-types&gt;` for the complete list of
        #     allowed types.
        # min_occurs: Defines the minimum number of times this ``field``
        #     may be used.  Its value is confined to zero or greater.
        #     Must be less than or equal to the value for the "maxOccurs"
        #     attribute.
        # recommended: A synonym for optional, but with the recommendation
        #     that this ``field`` be specified.
        # optional: A synonym for minOccurs=0.
        # max_occurs: Defines the maximum number of times this element may
        #     be used.  Its value is confined to zero or greater.  Must be
        #     greater than or equal to the value for the "minOccurs"
        #     attribute. A value of "unbounded" is allowed.
        # stride: The ``stride`` and ``data_offset`` attributes are used
        #     together to index the array of data items in a multi-
        #     dimensional array.  They may be used as an alternative
        #     method to address a data array that is not stored in the
        #     standard NeXus method of "C" order. The ``stride`` list
        #     chooses array locations from the data array  with each value
        #     in the ``stride`` list determining how many elements to move
        #     in each dimension. Setting a value in the ``stride`` array
        #     to 1 moves to each element in that dimension of the data
        #     array, while setting a value of 2 in a location in the
        #     ``stride`` array moves to every other element in that
        #     dimension of the data array.  A value in the ``stride`` list
        #     may be positive to move forward or negative to step
        #     backward. A value of zero will not step (and is of no
        #     particular use). See
        #     https://support.hdfgroup.org/HDF5/Tutor/phypereg.html or *4.
        #     Dataspace Selection Operations* in
        #     https://portal.hdfgroup.org/display/HDF5/Dataspaces The
        #     ``stride`` attribute contains a comma-separated list of
        #     integers. (In addition to the required comma delimiter,
        #     whitespace is also allowed to improve readability.) The
        #     number of items in the list is equal to the rank of the data
        #     being stored.  The value of each item is the spacing of the
        #     data items in that subscript of the array.
        # data_offset: The ``stride`` and ``data_offset`` attributes are
        #     used together to index the array of data items in a multi-
        #     dimensional array.  They may be used as an alternative
        #     method to address a data array that is not stored in the
        #     standard NeXus method of "C" order. The ``data_offset``
        #     attribute determines the starting coordinates of the data
        #     array for each dimension. See
        #     https://support.hdfgroup.org/HDF5/Tutor/phypereg.html or *4.
        #     Dataspace Selection Operations* in
        #     https://portal.hdfgroup.org/display/HDF5/Dataspaces The
        #     ``data_offset`` attribute contains a comma-separated list of
        #     integers. (In addition to the required comma delimiter,
        #     whitespace is also allowed to improve readability.) The
        #     number of items in the list is equal to the rank of the data
        #     being stored.  The value of each item is the offset in the
        #     array of the first data item of that subscript of the array.
        # interpretation: This instructs the consumer of the data what the
        #     last dimensions of the data are. It allows plotting software
        #     to work out the natural way of displaying the data. For
        #     example a single-element, energy-resolving, fluorescence
        #     detector with 512 bins should have
        #     ``interpretation="spectrum"``. If the detector is scanned
        #     over a 512 x 512 spatial grid, the data reported will be of
        #     dimensions: 512 x 512 x 512. In this example, the initial
        #     plotting representation should default to data of the same
        #     dimensions of a 512 x 512 pixel ``image`` detector where the
        #     images where taken at 512 different pressure values. In
        #     simple terms, the allowed values mean: * ``scalar`` = 0-D
        #     data to be plotted * ``scaler`` = DEPRECATED, use ``scalar``
        #     * ``spectrum`` = 1-D data to be plotted * ``image`` = 2-D
        #     data to be plotted * ``rgb-image`` = 3-D data to be plotted
        #     * ``rgba-image`` = 3-D data to be plotted * ``hsl-image`` =
        #     3-D data to be plotted * ``hsla-image`` = 3-D data to be
        #     plotted * ``cmyk-image`` = 3-D data to be plotted *
        #     ``vertex`` = 3-D data to be plotted

    # print("\n".join(output))
    # breakpoint()
