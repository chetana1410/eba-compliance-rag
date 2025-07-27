#!/usr/bin/env python3
import sys
from lxml import etree


def convert(xml_input: str, xslt_path: str) -> str:
    """Apply the XSLT transform and return LaTeX output."""
    dom = etree.parse(xml_input)
    xslt = etree.parse(xslt_path)
    transformer = etree.XSLT(xslt)
    result = transformer(dom)
    return str(result)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.xml> <transform.xsl>")
        sys.exit(1)

    input_xml = sys.argv[1]
    xslt_file = sys.argv[2]
    latex = convert(input_xml, xslt_file)
    print(latex)



