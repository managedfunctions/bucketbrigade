import arrow
import pandas as pd
import re

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from xml.etree import ElementTree as ET

from pydantic import (
    BaseModel,
    computed_field,
    parse_obj_as,
    ValidationError,
)
from typing import Any, Type, Optional

arrow.now().isoformat()


def snake(input_str: str, ignore_dot: bool = False) -> str:
    if ignore_dot:
        trans_table = str.maketrans(" -", "__")
    else:
        trans_table = str.maketrans(" .-", "___")

    input_str = input_str.translate(trans_table)
    input_str = re.sub("_+", "_", input_str)

    input_str = re.sub(r"[^\w_.]" if ignore_dot else r"[^\w_]", "", input_str)

    input_str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", input_str).lower()

    input_str = re.sub("_+", "_", input_str)

    return input_str.strip("_").strip()


class Metadata(BaseModel):
    bucket: str
    pipe: str
    folder: str
    partner_name: Optional[str] = ""
    environment: str = "prod"
    timezone: str = "Australia/Sydney"

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    def bucket_path(self) -> str:
        return f"s3://{self.bucket}"

    @computed_field
    def pipe_path(self) -> str:
        return f"s3://{self.bucket}/{self.pipe}"

    @computed_field
    def folder_path(self) -> str:
        return f"{self.bucket_path}/{self.pipe}/{self.folder}/{self.environment}"

    @computed_field
    def input_path(self) -> str:
        return f"{self.folder_path}/input_folder"

    @computed_field
    def archive_path(self) -> str:
        return f"{self.folder_path}/archive_folder"

    @computed_field
    def output_path(self) -> str:
        return f"{self.folder_path}/output_folder"

    @computed_field
    def skip_path(self) -> str:
        return f"{self.folder_path}/skip_folder"

    @computed_field
    def aggregate_path(self) -> str:
        return f"{self.folder_path}/aggregate_folder"

    @computed_field
    def map_path(self) -> str:
        return f"{self.folder_path}/map_folder"

    @computed_field
    def today(self) -> str:
        return arrow.now(self.timezone).format("YYYY-MM-DD")

    @computed_field
    def now(self) -> str:
        return arrow.now(self.timezone).isoformat()

    @computed_field
    def now_prefix(self) -> str:
        return arrow.now(self.timezone).format("YYYYMMDDHHmmss")


def get_metadata_from_docpath(docpath):
    docpath_parts = docpath.split("//")[-1].split("/")
    return Metadata(
        bucket=docpath_parts[0],
        pipe=docpath_parts[1],
        folder=docpath_parts[2],
        environment=docpath_parts[3],
    )

def validate_df(
    df: pd.DataFrame,
    model: Type[BaseModel],
    drop_missing=True,
    return_errors=False,
    print_errors=True,
) -> pd.DataFrame:
    # Extract aliases and rename DataFrame columns based on aliases
    field_to_alias = {
        field_name: field.alias
        for field_name, field in model.__annotations__.items()
        if hasattr(field, "alias")
    }
    df = df.rename(columns={v: k for k, v in field_to_alias.items()})

    # Validate and collect errors
    validated_rows, error_messages = [], []
    for idx, row in df.iterrows():
        try:
            instance = parse_obj_as(model, row.to_dict())
            validated_rows.append(instance.dict())
        except ValidationError as e:
            e = re.sub(
                r"For further information visit https://errors.pydantic.dev/\d+\.\d+/v/[a-z_]+",
                "",
                str(e),
            )
            error_messages.append((idx, e))
            validated_rows.append(
                row.to_dict()
            )  # Append the original row if validation fails

    # Print errors if requested
    if print_errors and error_messages:
        for idx, error in error_messages:
            print(f"Row {idx}: {error}")

    # Create a new DataFrame for validated rows
    validated_df = pd.DataFrame(validated_rows)
    valid_columns = [
        col for col in model.__annotations__.keys() if col in validated_df.columns
    ]

    if drop_missing:
        validated_df = validated_df.loc[:, valid_columns]

    # Add error messages column
    if return_errors:
        error_col_name = "validation_errors"
        counter = 0
        while error_col_name in validated_df.columns:
            counter += 1
            error_col_name = f"validation_errors_{str(counter).zfill(2)}"
        validated_df[error_col_name] = [None] * len(validated_df)
        for idx, error in error_messages:
            validated_df.at[idx, error_col_name] = str(error)

    return validated_df


def model_instance_to_dataframe(model_instance: BaseModel) -> pd.DataFrame:
    data = {
        key: value
        for key, value in model_instance.model_dump().items()
        if not callable(value)
    }
    return pd.DataFrame.from_dict(data, orient="index")



def merge_dicts(dict1, dict2, suffix="_y"):
    """
    Merges two dictionaries. If there are duplicate keys, appends a suffix to the keys from the second dictionary.

    :param dict1: The first dictionary.
    :param dict2: The second dictionary.
    :param suffix: The suffix to append to duplicate keys from dict2. Defaults to '_y'.
    :return: A new dictionary with merged contents.
    """
    merged_dict = dict1.copy()  # Start with the contents of dict1

    # Loop through dict2, modifying and adding to the new dictionary
    for key, value in dict2.items():
        new_key = key + suffix if key in dict1 else key
        merged_dict[new_key] = value

    return merged_dict


def explode_df_on_column(df, column):
    df_exploded = df.explode(column)
    df_exploded_reset = df_exploded.reset_index().rename(
        columns={"index": "original_index"}
    )
    df_lines = pd.json_normalize(df_exploded_reset[column])
    df_result = df_exploded_reset.join(df_lines)
    df_result = df_result.drop(columns=[column, "original_index"])
    df_result.columns = [snake(x) for x in df_result.columns]
    return df_result



def rx(n, x=2):
    """Round a number to a specified number of decimal places with ROUND_HALF_UP."""
    try:
        # Create a string representing the format, e.g., '0.00' for 2 decimal places
        format_str = f"0.{'0' * x}"
        # Ensure n is a Decimal, which can handle float, int, or string input
        result = Decimal(str(n)).quantize(Decimal(format_str), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        # Handle cases where n is None or not a valid number
        raise ValueError(f"Invalid input for decimal conversion: {n}")
    return result


def Rxp(decimal_places: int):
    class RoundedDecimal(Decimal):
        @classmethod
        def __get_validators__(cls):
            yield cls.validate

        @classmethod
        def validate(cls, v: Any) -> Decimal:
            if not isinstance(v, (float, int, Decimal)):
                raise ValueError(f"Value {v} is not a valid decimal/float/int")
            format_str = f"0.{'0' * decimal_places}"
            return Decimal(str(v)).quantize(Decimal(format_str), rounding=ROUND_HALF_UP)

        @classmethod
        def __get_pydantic_core_schema__(cls, **kwargs):
            return {
                "type": "number",
                "format": "decimal",
                "decimal_places": decimal_places,
            }

    return RoundedDecimal


def remove_xml_namespace_prefixes(xml_string):
    # Parse the XML string
    root = ET.fromstring(xml_string)

    # Iterate through all elements and remove namespace prefixes
    for elem in root.iter():
        if ":" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        if elem.attrib:
            elem.attrib = {k.split("}")[-1]: v for k, v in elem.attrib.items()}

    # Convert the modified XML back to a string
    return ET.tostring(root, encoding="unicode")


def clean_key(key):
    """Clean unwanted characters from XML keys."""
    return key.replace("@", "").replace("#", "")


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
                d[t.tag]["text"] = text
        else:
            d[t.tag] = text
    return d


def parse_xml(xml_string):
    """Parse XML string into dictionary."""
    xml_string = remove_xml_namespace_prefixes(xml_string)
    tree = ET.ElementTree(ET.fromstring(xml_string))
    return etree_to_dict(tree.getroot())


def flatten_dict(dictionary, parent_key="", sep="_", snake_keys=True):
    items = {}
    for k, v in dictionary.items():
        if snake_keys:
            k = snake(k)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                list_key = f"{new_key}{sep}{idx}"
                if isinstance(item, (dict, list)):
                    items.update(flatten_dict(item, list_key, sep=sep))
                else:
                    items[list_key] = item
        else:
            items[new_key] = v
    return items


