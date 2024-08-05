import sys
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path
import glob
from xsdata_pydantic.bindings import XmlParser

from .nxdl import Definition

# NS = "{http://definition.nexusformat.org/nxdl/3.1}"


# class Attribute(BaseModel):
#     name: str
#     datatype: str = "NXChar"
#     recommended: bool = False
#     optional: bool = True
#     doc: str | None = None
#     enumeration: list[Any] = []
#     dimensions: list[Any] = []

#     @classmethod
#     def parse(cls, root: ElementTree.Element, clear: bool = False) -> Self:
#         attr = Attribute(
#             name=root.attrib["name"],
#         )
#         if clear:
#             del root.attrib["name"]
#             assert not root.attrib


# class BaseDefinition(BaseModel):
#     name: str
#     parent: str
#     doc: str
#     attributes: list[Attribute]

#     @classmethod
#     def parse(cls, root: ElementTree.Element, clear: bool = False) -> Self:
#         parts = {
#             "name": root.attrib["name"],
#             "parent": root.attrib["extends"],
#         }

#         if root.attrib["type"] != "group" or root.attrib["category"] != "base":
#             raise RuntimeError("Group or type does not match")
#         if clear:
#             del root.attrib["{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"]
#             del root.attrib["category"]
#             del root.attrib["extends"]
#             del root.attrib["type"]
#             del root.attrib["name"]
#             assert not root.attrib, "Failed to clear root properties"

#         # Now, four classes of child:
#         # - doc: Documentation for this object
#         # - attribute: Something held as an hdf5 attribute
#         # - group: A subgroup of NX objects
#         # - field: A dataset that holds a value in this object
#         if doc := root.find("{*}doc}"):
#             parts["doc"] = textwrap.dedent(doc.text).strip()
#             root.remove(doc)

#         parts["attributes"] = [Attribute.parse(x) for x in root.findall("{*}attribute")]

#         return BaseDefinition(**parts)


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "sources",
        nargs="*",
        default=["v2024.2/base_classes/NXentry.nxdl.xml"],
    )
    args = parser.parse_args()

    # Expand every passed file as if it contained a glob
    args.sources = list(
        chain(*[[Path(x) for x in glob.glob(p, recursive=True)] for p in args.sources])
    )
    if not args.sources:
        sys.exit("Error: No sources found after expansion")

    # print(args)
    base_definitions: list[Definition] = []
    for source in args.sources:
        parser = XmlParser()
        defn = parser.parse(source, Definition)
