import sys
from typing import Self
from xml.etree import ElementTree
from argparse import ArgumentParser
from itertools import chain
from pydantic import BaseModel
from pathlib import Path
import glob

# NS = "{http://definition.nexusformat.org/nxdl/3.1}"


class BaseDefinition(BaseModel):
    name: str
    parent: str

    @classmethod
    def parse(cls, root: ElementTree.Element, clear: bool = False) -> Self:
        obj = BaseDefinition(name=root.attrib["name"], parent=root.attrib["extends"])

        if clear:
            del root.attrib["{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"]
            del root.attrib["category"]
            del root.attrib["extends"]
            del root.attrib["type"]
            del root.attrib["name"]

        # {'name': 'NXentry', 'type': 'group', 'extends': 'NXobject', 'category': 'base', '{http://www.w3.org/2001/XMLSchema-instance}schemaLocation': 'http://definition.nexusformat.org/nxdl/3.1 ../nxdl.xsd'}


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
    base_definitions: list[BaseDefinition] = []
    for source in args.sources:
        tree = ElementTree.parse(source)
        base_definitions.append(BaseDefinition.parse(tree.getroot(), clear=True))

    breakpoint()
