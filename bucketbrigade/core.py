import arrow
import re

from xml.etree import ElementTree as ET

from pydantic import (
    BaseModel,
    computed_field,
)

def snake(input_str: str, ignore_dot: bool = False) -> str:
    if ignore_dot:
        trans_table = str.maketrans(' -', '__')
    else:
        trans_table = str.maketrans(' .-', '___')

    input_str = input_str.translate(trans_table)
    input_str = re.sub('_+', '_', input_str)

    input_str = re.sub(r'[^\w_.]' if ignore_dot else r'[^\w_]', '', input_str)

    input_str = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', input_str).lower()

    input_str = re.sub('_+', '_', input_str)

    return input_str.strip('_').strip()

def remove_xml_namespace_prefixes(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Iterate through all elements and remove namespace prefixes
    for elem in root.iter():
        if ':' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        if elem.attrib:
            elem.attrib = {k.split('}')[-1]: v for k, v in elem.attrib.items()}

    # Convert the modified XML back to a string
    return ET.tostring(root, encoding="unicode")


def clean_key(key):
    """Clean unwanted characters from XML keys."""
    return key.replace('@', '').replace('#', '')

def etree_to_dict(t):
    """Recursively convert ElementTree into dictionary."""
    d = {clean_key(t.tag): {} if t.attrib else None}
    children = list(t)
    if children:
        dd = {}
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                if k not in dd:
                    dd[k] = v
                else:
                    if not isinstance(dd[k], list):
                        dd[k] = [dd[k]]
                    dd[k].append(v)
        d = {clean_key(t.tag): dd}
    if t.attrib:
        d[t.tag].update((clean_key(k), v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['text'] = text
        else:
            d[t.tag] = text
    return d


def parse_xml(xml_string):
    """Parse XML string into dictionary."""
    xml_string = remove_xml_namespace_prefixes(xml_string)
    tree = ET.ElementTree(ET.fromstring(xml_string))
    return etree_to_dict(tree.getroot())