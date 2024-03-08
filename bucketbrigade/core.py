import json
import os
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Type
from xml.etree import ElementTree as ET

import arrow
import dateparser
import duckdb
import modal
import pandas as pd
from unidecode import unidecode
from dateutil import tz
from dopplersdk import DopplerSDK
from IPython.display import display
from magika import Magika
from pydantic import (
    BaseModel,
    ValidationError,
    computed_field,
    parse_obj_as,
)
from rank_bm25 import BM25L
from yaml import Loader, load

m = Magika()

if modal.is_local():
    temp_path = Path("./tmp")
    temp_path.mkdir(parents=True, exist_ok=True)
    function_location = "local"
else:
    temp_path = Path("/root/tmp")
    function_location = "remote"


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
    direction: str
    system: str
    object: str
    variant: int
    dimension: Optional[str] = ""
    environment: str = "prod"
    timezone: str = "Australia/Sydney"

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    def folder(self) -> str:
        return f"{self.object}_variant_{self.variant}".replace(" ", "")

    @computed_field
    def function(self) -> str:
        return f"{self.direction}_{self.system}_{self.folder}".replace(" ", "")

    @computed_field
    def bucket_path(self) -> str:
        return f"s3://{self.bucket}".replace(" ", "")

    @computed_field
    def system_path(self) -> str:
        return f"s3://{self.bucket}/{self.direction}_{self.system}".replace(" ", "")

    @computed_field
    def folder_path(self) -> str:
        return f"{self.system_path}/{self.folder}/{self.environment}".replace(" ", "")

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


def get_metadata_from_docpath(docpath, dimension=""):
    docpath_parts = docpath.split("//")[-1].split("/")
    return Metadata(
        bucket=docpath_parts[0],
        direction=docpath_parts[1].split("_")[0],
        system=docpath_parts[1].rsplit("_", 1)[-1],
        object=docpath_parts[2].split("_variant_")[0],
        variant=docpath_parts[2].split("_variant_")[-1],
        dimension=dimension,
        environment=docpath_parts[3],
    )


# def get_metadata_from_variant(bucket, variant, environment="prod"):
#     variant_parts = variant.split("/")
#     return Metadata(
#         bucket=bucket,
#         direction=variant_parts[1].split("_")[0],
#         system=variant_parts[1].rsplit("_", 1)[-1],
#         folder=variant_parts[2],
#         environment=environment,
#     )


def setup_modal_uv_image(stub_name):
    sdk = set_doppler()
    github_token = sdk.secrets.get(project="github", config="prod", name="GITHUB_TOKEN")
    github_token = vars(github_token)["value"]["computed"]

    image = (
        modal.Image.debian_slim()
        .run_commands(
            [
                "apt-get update",
                "apt-get install -y git",
            ]
        )
        .apt_install("git", "default-libmysqlclient-dev")
        .pip_install("uv")
        .run_commands(
            [
                """uv pip install --compile "bucketbrigade @ git+https://www.github.com/managedfunctions/bucketbrigade.git" --system""",
                "force_build=True",
            ]
        )
    )

    stub = modal.Stub(
        name=stub_name,
    )
    return image, stub


def setup_modal_image(stub_name, force_build=False):
    sdk = set_doppler()
    github_token = sdk.secrets.get(project="github", config="prod", name="GITHUB_TOKEN")
    github_token = vars(github_token)["value"]["computed"]

    image = (
        modal.Image.debian_slim()
        .run_commands(
            [
                "apt-get update",
                "apt-get install -y git",
            ]
        )
        .apt_install("git", "default-libmysqlclient-dev")
        .pip_install(
            "git+https://www.github.com/managedfunctions/bucketbrigade.git",
            "ipython",
            "doppler-sdk",
            "magika",
            force_build=force_build,
        )
    )

    stub = modal.Stub(
        name=stub_name,
    )
    return image, stub


def setup_function_from_metadata(
    metadata: Metadata,
    secrets_config="",
    dimension="",
    secrets_provider="doppler",
    use_lowercase=False,
):
    secrets = get_secrets(
        metadata,
        config=secrets_config,
        dimension=dimension,
        provider=secrets_provider,
        use_lowercase=use_lowercase,
    )
    return metadata, secrets


def setup_function_from_docpath(
    docpath: str = "",
    include_string: str = "",
    path_number=0,
    secrets_config="",
    dimension="",
    secrets_provider="doppler",
    use_lowercase=False,
    cloud="",
):
    metadata = get_metadata_from_docpath(docpath)
    secrets = get_secrets(
        metadata,
        config=secrets_config,
        dimension=dimension,
        provider=secrets_provider,
        use_lowercase=use_lowercase,
    )
    if modal.is_local():
        docpaths_to_process = cloud.get_docpaths_to_process(
            metadata.input_path, include_string=include_string
        )
        if len(docpaths_to_process) == 0:
            print("\nNo docpaths to process.\n")
            return metadata, secrets, None
        elif len(docpaths_to_process) > 0:
            docs_to_show = min(5, len(docpaths_to_process))
            if len(docpaths_to_process) > 1:
                print(
                    f"\n{len(docpaths_to_process)} docpaths to process. Here are the first {docs_to_show}:\n"
                )
            else:
                print(f"\n{len(docpaths_to_process)} docpath to process.\n")
            for i, docpath in enumerate(docpaths_to_process[:docs_to_show]):
                docname = docpath.split("/")[-1]
                print(i + 1, docname)
        docpath = docpaths_to_process[path_number]
    return metadata, secrets, docpath


def set_doppler(provider_key=""):
    # Initialize the Doppler SDK
    doppler = DopplerSDK()

    # Determine the access token: first try the parameter, then the environment
    access_token = provider_key if provider_key else os.getenv("DOPPLER_TOKEN")

    if access_token:
        doppler.set_access_token(access_token)
    else:
        raise ValueError(
            "Doppler access token not provided and DOPPLER_TOKEN environment variable is not set."
        )
    return doppler


def get_functions(
    function, environment="prod", dimension="", provider="", project_prefix="mfs-"
):
    if not provider or provider == "doppler":
        sdk = set_doppler()
    results = sdk.projects.list(page=1, per_page=100)

    all_configs = []
    relevant_configs = []
    projects = vars(results)["projects"]
    projects = [
        project["slug"]
        for project in projects
        if project["slug"].startswith(project_prefix)
    ]
    for project in projects:
        configs = sdk.configs.list(project=project)
        configs = vars(configs)["configs"]
        for config in configs:
            if config["name"].startswith(environment) and config["name"] != environment:
                system = "_".join(config["name"].split("_")[1:])
                config_object = {
                    "project": project,
                    "system": system,
                    "environment": environment,
                }
                all_configs.append(config_object)
                if function in config["name"]:
                    print(project, "->", config["name"])
                    relevant_configs.append(config_object)
    return relevant_configs


def get_folder_from_variant(row):
    folder = row.variants.split("/")[-1]
    folder = folder.split(f"{row.direction}_{row.system}")[-1]
    folder = folder.strip("_/")
    folder = folder.split("_variant_")[0]
    return folder


def setup_project_db(cloud, provider="", project_prefix="mfs-"):
    if not provider or provider == "doppler":
        sdk = set_doppler()
    results = sdk.projects.list(page=1, per_page=100)
    environments = ["prod", "test"]
    directions = ["from", "to", "in"]
    all_configs = []
    projects = vars(results)["projects"]
    projects = [
        project["slug"]
        for project in projects
        if project["slug"].startswith(project_prefix)
    ]
    for project in projects:
        print(project)
        configs = sdk.configs.list(project=project)
        configs = vars(configs)["configs"]
        for config in configs:
            if config["name"] not in environments:
                system = "_".join(config["name"].split("_")[1:])
                environment = config["name"].split("_")[0]
                config_object = {
                    "project": project,
                    "system": system,
                    "environment": environment,
                }
                variants = sdk.secrets.get(
                    project=config_object["project"],
                    config=f'{config_object["environment"]}_{config_object["system"]}',
                    name="VARIANTS",
                )
                variants = vars(variants)["value"]["computed"]
                if variants:
                    variants = load(variants, Loader=Loader)
                else:
                    variants = []
                config_object["variants"] = variants
                all_configs.append(config_object)
    df = pd.DataFrame(all_configs)
    df = df.explode("variants")
    df.variants = df.variants.fillna("")
    df["function_type"] = df.variants.apply(
        lambda x: "to_map" if "/" in x else "to_api"
    )
    df["direction"] = df.variants.apply(
        lambda x: x.split("_")[0] if x.split("_")[0] in directions else ""
    )
    df["folder"] = df.apply(get_folder_from_variant, axis=1)
    cloud.save_doc("s3://mfs-admin/configs/project_db.parquet", df, dated=False)
    return df


def get_functions_to_process_by_system(system, environment, cloud):
    df = pd.read_parquet("s3://mfs-admin/configs/project_db.parquet")
    df = df[(df.system == system) & (df.environment == environment)]
    df = df[df.variants != ""]
    return df[df.function_type == "to_map"], df[df.function_type == "to_api"]


def get_functions_to_process_by_customer(customer, environment, cloud):
    df = pd.read_parquet("s3://mfs-admin/configs/project_db.parquet")
    df = df[(df.project == customer) & (df.environment == environment)]
    df = df[df.variants != ""]
    return df[df.function_type == "to_map"], df[df.function_type == "to_api"]


def process_all_folders(df_to_map, functions, cloud):
    dps = []  # to do: load up dps with everything to process and the run the map function
    for k, row in df_to_map.iterrows():
        folder_path = f"s3://{row.project}/{row.variants}/{row.environment}"
        print(folder_path)
        metadata = get_metadata_from_docpath(folder_path)
        docpaths_to_process = cloud.get_docpaths_to_process(metadata.input_path)
        process_function = getattr(functions, metadata.function)
        errors = list(process_function.map(docpaths_to_process, return_exceptions=True))
    return errors


def process_all_apis(df_to_api, functions, cloud):
    for k, row in df_to_api.iterrows():
        process_function = getattr(functions, row.variants)
        print(process_function)
        print(row)
        metadata = Metadata(
            bucket=row.project,
            direction=row.direction,
            system=row.system,
            folder=row.folder,
            environment=row.environment,
            partner_name="",
        )
        process_function.local(metadata)


def convert_to_dict(v):
    try:
        v = load(v, Loader=Loader)
    except:
        try:
            v = json.loads(v)
        except:
            pass
    return v


def update_values_by_dimension(data: dict, dimension: str) -> dict:
    for key, value in data.items():
        if isinstance(value, dict) and dimension in value:
            data[key] = value[dimension]
    return data


def get_secrets(
    metadata,
    config="",
    provider="doppler",
    provider_key=None,
    use_lowercase=True,
):
    if provider == "doppler":
        sdk = set_doppler()
    # Build the config from metadata.system and metadata.environment
    if not config:
        config = f"{metadata.environment}_{metadata.system}"

    # Fetch secrets for the specified project and config
    try:
        results = sdk.secrets.list(project=metadata.bucket, config=config)
    except Exception as e:
        print(e)
        print()
        print(json.loads(e.message)["messages"][0])
        print()
        try:
            results = sdk.secrets.list(
                project=metadata.bucket, config=metadata.environment
            )
        except Exception as e:
            print()
            print(json.loads(e.message)["messages"][0])
            print()
            print("Available configs for this project:")
            config_list = vars(sdk.configs.list(project=metadata.bucket))["configs"]
            print()
            for v in config_list:
                print(" -", v["name"])
            return ""

    rds_secrets = vars(results)["secrets"]

    # Process secrets: filter out keys containing "DOPPLER" and adjust case based on flag
    if use_lowercase:
        processed_secrets = {
            k.lower(): convert_to_dict(v["computed"])
            for k, v in rds_secrets.items()
            if "DOPPLER" not in k
        }
    else:
        processed_secrets = {
            k.upper(): convert_to_dict(v["computed"])
            for k, v in rds_secrets.items()
            if "DOPPLER" not in k
        }
    if metadata.dimension:
        processed_secrets = update_values_by_dimension(
            processed_secrets, metadata.dimension
        )
    return processed_secrets


def validate_df_rows(df: pd.DataFrame, model_class) -> (pd.DataFrame, list):
    """Validates each row of a DataFrame against a Pydantic model.

    Args:
        df: The DataFrame to validate.
        model_class: The Pydantic model class to use for validation.

    Returns:
        A tuple of a DataFrame containing only valid rows, and a list of dictionaries with error details for invalid rows.
    """
    valid_rows = []
    error_details = []

    for index, row in df.iterrows():
        try:
            model_instance = model_class(**row.to_dict())
            valid_rows.append(row.to_dict())  # If validation succeeds, keep the row
        except TypeError as e:
            # TypeError occurs when a value is missing from the row
            error_info = {
                "index": index,
                "row_data": row.to_dict(),
                "errors": [{"loc": "N/A", "msg": str(e), "type": "TypeError"}],
            }
            error_details.append(error_info)
            # Optionally, print each error as it occurs
            print(f"Validation error at index {index}:", str(e))
            display(row.to_dict())
            return ""
        except ValidationError as e:
            # Aggregate detailed error information for each invalid row
            errors = e.errors()
            detailed_errors = [
                {"loc": err["loc"][0], "msg": err["msg"], "type": err["type"]}
                for err in errors
            ]
            error_info = {
                "index": index,
                "row_data": row.to_dict(),
                "errors": detailed_errors,
            }
            error_details.append(error_info)
            # Optionally, print each error as it occurs
            print(f"Validation error at index {index}:", detailed_errors)
            display(row.to_dict())
            return ""
    valid_rows_df = pd.DataFrame(valid_rows)

    # Return the DataFrame of valid rows and the list of error details
    return valid_rows_df, error_details


def to_caps_case_class_name(name):
    # Example function to convert function name to CapsCase for class name
    # Adjust this according to your actual naming convention if necessary
    return "".join(word.capitalize() for word in name.split("_"))


def validate_function_output(model_class=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if model_class and isinstance(result, pd.DataFrame):
                print()
                print(f"Output from function: {func.__name__}")
                print()

                # Display head if it's still a DataFrame
                display(result.head())

                try:
                    # Assuming model_class is a Pydantic model class
                    result = validate_df_rows(result, model_class)
                    print(
                        f"DataFrame validated successfully against {model_class.__name__}."
                    )
                    print()
                except ValidationError as e:
                    print(f"Validation errors:\n{e}")
                    print()
            else:
                print(
                    "No model class provided or result is not a DataFrame. Skipping validation."
                )
                print()

            return result

        return wrapper

    return decorator


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

        display(validated_df)
    # else:
    #     if error_messages:
    #         raise ValueError(f"Validation errors: {error_messages}")

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


def clean_text(data, remove_line_breaks=True, unidecoded=True):
    if isinstance(data, dict):
        data = {
            key: clean_text(value, remove_line_breaks=remove_line_breaks)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        data = [
            clean_text(item, remove_line_breaks=remove_line_breaks) for item in data
        ]
    else:
        if remove_line_breaks is True:
            data = re.sub(" +", " ", " ".join(data.splitlines()))
        if unidecoded:
            data = unidecode(data)
        data = re.sub("\n+", "\n", data).strip()
    return data


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


def get_duckdb_connection(aws_profile=None):
    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute("INSTALL 'httpfs'")
    con.execute("INSTALL 'aws'")
    con.execute("LOAD 'httpfs'")
    con.execute("LOAD 'aws'")
    # Optionally use a specific AWS profile
    if aws_profile:
        con.execute(f"CALL load_aws_credentials('{aws_profile}')")
    else:
        con.execute("CALL load_aws_credentials()")
    return con


def get_db_table(s3_path, table_name="db", aws_profile=None):
    try:
        con = get_duckdb_connection(aws_profile=aws_profile)
        # Directly create a DuckDB table from the Parquet file
        con.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{s3_path}')"
        )
        # Retrieve the DataFrame for use in Python, if needed
        return con
    except Exception as e:
        print(f"Error loading table from {s3_path}: {e}")
        return None


def read_db_table(con):
    df = con.table("db").df()
    df = remove_index_column(df)
    return df


def get_duckdb_schema(con):
    # Retrieve the list of tables
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()
    schemas = []

    # For each table, retrieve column details and format them
    for table in tables:
        table_name = table[0]
        columns_info = con.execute(
            f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'main'"
        ).fetchall()
        columns_str = ",\n    ".join([f"{col[0]} {col[1]}" for col in columns_info])
        table_schema = f"CREATE TABLE {table_name} (\n    {columns_str}\n);\n"
        schemas.append(table_schema)

    # Combine all table schemas into one formatted string
    full_schema = "\n".join(schemas)
    return full_schema


def remove_index_column(df):
    if "index" in df.columns:
        if df["index"].to_list() == df.index.to_list():
            df = df.drop(columns=["index"])
    return df


def rollup_df(model_name, documents_path, metadata, cloud):
    con = get_db_table(documents_path)
    df = con.table("db").df()
    print(f"Original table shape: {df.shape}")
    mp = Path.cwd() / f"models/{model_name}.sql"
    df_rollup = con.execute(mp.read_text()).fetchdf()
    df_rollup = remove_index_column(df_rollup)
    print(f"New table shape: {df_rollup.shape}")
    df_path = f"{metadata.aggregate_path}/{model_name}.parquet"
    cloud.save_doc(df_path, df_rollup, dated=False)
    return df_path


def convert_to_money(line: str) -> float:
    """
    Converts the first monetary string to a float, treating CR, DR, CF, numbers preceded by '-', and parentheses as negatives.
    """
    # Define pattern to capture monetary values including optional CR, DR, CF indicators or parentheses for negative values.
    pattern = re.compile(r"-?\$?[\d,]+(?:\.\d+)?(?:CR|DR|CF)?|\(.*?\)")
    line = f"{line}"
    match = pattern.search(line)

    if match:
        value = match.group()
        # Determine if the value is negative based on CR, DR, CF, parentheses, or preceding '-'.
        is_negative = (
            "CR" in value
            or "DR" in value
            or "CF" in value
            or "(" in value
            or value.startswith("-")
        )

        # Clean the string from any non-numeric characters for conversion.
        num = float(re.sub(r"[^\d.]", "", value))

        # Apply negative multiplier if any negative indicators are found.
        if is_negative:
            num = -abs(num)

        return num
    else:
        # Return 0.0 if no match is found.
        return 0.0


def get_date(
    text_or_datetime,
    month_first: bool = False,
    preferred_timezone: str = "Australia/Sydney",
) -> str:
    """
    Extracts the date from a text string or datetime object, handling different date formats and timezone conversions.

    Parameters:
    - text_or_datetime (str | datetime): Input text containing a date or a datetime object.
    - month_first (bool): Determines if the month comes before the day in ambiguous date formats. Defaults to True.
    - preferred_timezone (str): Preferred timezone for converting datetime objects. Defaults to 'UTC'.

    Returns:
    - str: The date in ISO format (YYYY-MM-DD) if a valid date is found or provided.
    """
    if isinstance(text_or_datetime, datetime):
        # Convert datetime to the preferred timezone if it's timezone-aware.
        if text_or_datetime.tzinfo:
            target_timezone = tz.gettz(preferred_timezone)
            text_or_datetime = text_or_datetime.astimezone(target_timezone)
        return text_or_datetime.strftime("%Y-%m-%d")
    else:
        # Define a simple date pattern.
        date_pattern = re.compile(
            r"\d{4}-\d{1,2}-\d{1,2}|\d{2}/\d{2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}"  # Adjust pattern as needed
        )
        # Find all matches based on the pattern.
        matches = date_pattern.findall(text_or_datetime)

        if matches:
            # Directly use the month_first parameter with dateparser.
            date = dateparser.parse(
                matches[0],
                settings={
                    "DATE_ORDER": "MDY" if month_first else "DMY",
                    "PREFER_DATES_FROM": "past",
                    "TIMEZONE": preferred_timezone,
                },
            )

            if date:
                # Ensure the date is in the preferred timezone if it's timezone-aware.
                if date.tzinfo:
                    target_timezone = tz.gettz(preferred_timezone)
                    date = date.astimezone(target_timezone)
                return date.strftime("%Y-%m-%d")
            else:
                return "No valid date found."
        else:
            return "No valid date found."


def make_ngrams(text):
    text = split_alpha_numeric(re.sub(r"[-_]", " ", text.lower()))
    ngram_sets = [
        {
            "".join(t)
            for i in range(n)
            for t in zip(*[text[j:] for j in range(i, i + n)])
        }
        for n in [3, 5, 7, 9, 11, 13]
    ]
    return list(
        set.union(
            *ngram_sets,
            {text},
            set(re.findall(r"[A-Za-z]+", text)),
            set(re.findall(r"[0-9\.\-,]{2,}", text)),
        )
    )


def split_alpha_numeric(s):
    return " ".join(re.split(r"([0-9]+)", s))


def get_bm25_scores(query, corpus, substring_promotion=20):
    if isinstance(corpus, pd.Series):
        corpus = corpus.tolist()

    cleaned_query = split_alpha_numeric(re.sub(r"[^a-zA-Z0-9 ]+", "", query.lower()))
    tokenized_query = make_ngrams(query)

    cleaned_corpus = [
        split_alpha_numeric(re.sub(r"[^a-zA-Z0-9 ]+", "", doc.lower()))
        for doc in corpus
    ]
    tokenized_corpus = [make_ngrams(doc) for doc in corpus]

    bm25_obj = BM25L(tokenized_corpus) if tokenized_corpus else None

    if bm25_obj is None:
        return []

    scores = bm25_obj.get_scores(tokenized_query)

    # Identify substring matches
    if len(cleaned_query) > 6:
        mask = [
            cleaned_query in doc or (doc and doc in cleaned_query)
            for doc in cleaned_corpus
        ]
    else:
        mask = [False] * len(cleaned_corpus)

    return [
        score + (substring_promotion if match else 0)
        for score, match in zip(scores, mask)
    ]


def group_and_sum(df, groupby_col, sum_col):
    """
    Groups the dataframe by the specified column and sums another specified column,
    while retaining all other columns.

    :param df: DataFrame to process.
    :param groupby_col: Column name to group by.
    :param sum_col: Column name to sum.
    :return: Grouped and summed DataFrame.
    """
    # Group by the specified column and sum the specified column
    grouped_sum = df.groupby(groupby_col, as_index=False)[sum_col].sum()

    # Dropping the sum_col from the original dataframe to avoid duplication
    df_dropped = df.drop(columns=[sum_col])

    # Removing duplicates based on the groupby_col
    df_dropped = df_dropped.drop_duplicates(subset=groupby_col)

    # Merging the grouped_sum dataframe with the df_dropped dataframe
    result_df = pd.merge(grouped_sum, df_dropped, on=groupby_col)

    return result_df


# Code to work on that converts json to dataframes, creates pydantic models and the validates them:


# from typing import Any, Dict, List
# from pydantic import BaseModel, create_model, Field, ValidationError
# import pandas as pd
# from pandas import json_normalize
# from datetime import datetime
# import json

# def flatten_json_to_df(json_obj: List[Dict], max_levels: int = 1) -> pd.DataFrame:
#     return json_normalize(json_obj, max_level=max_levels)

# def generate_pydantic_model(df: pd.DataFrame) -> BaseModel:
#     fields = {}
#     for column in df.columns:
#         dtype = df[column].dtype
#         if pd.api.types.is_integer_dtype(dtype):
#             pydantic_type = int
#         elif pd.api.types.is_float_dtype(dtype):
#             pydantic_type = float
#         elif pd.api.types.is_bool_dtype(dtype):
#             pydantic_type = bool
#         elif pd.api.types.is_datetime64_any_dtype(dtype):
#             pydantic_type = datetime
#         else:
#             # Convert complex structures to JSON strings to avoid serialization issues
#             pydantic_type = str
#             # Convert column to string if it contains complex data
#             df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

#         fields[column] = (pydantic_type, ...)

#     return create_model('DynamicModel', **fields)

# def validate_df(df: pd.DataFrame, model: BaseModel):
#     dfs = []
#     for index, row in df.iterrows():
#         try:
#             model_obj = model(**row.to_dict())
#             dfs.append(model_obj.model_dump())
#             print(f"Row {index} validation successful: {model_obj}")
#         except ValidationError as e:
#             print(f"Validation error in row {index}: {e.json()}")
#             return False
#     return pd.DataFrame(dfs)

# # Example usage
# if __name__ == "__main__":
#     data = [{
#         "id": 1,
#         "name": "Alice",
#         "details": {
#             "age": 30,
#             "interests": ["cycling", "hiking"],
#             "dogs": {"name": "Fido", "breed": "Labrador"}
#         },
#         "balance": 100.5,
#         "created_at": "2021-01-01T00:00:00"
#     }]

#     df = flatten_json_to_df(data, max_levels=2)

#     # Convert 'created_at' column to datetime
#     df['created_at'] = pd.to_datetime(df['created_at'])
#     display(df)

#     DynamicModel = generate_pydantic_model(df)

#     # Validate DataFrame rows against the Pydantic model
#     validate_df(df, DynamicModel)
