import arrow
import pandas as pd
import re
from pydantic import (
    BaseModel,
    computed_field,
    parse_obj_as,
    ValidationError,
)
from typing import Optional, Type

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

def validate_df(df: pd.DataFrame, model: Type[BaseModel], drop_missing=True, return_errors=False, print_errors=True) -> pd.DataFrame:
    # Extract aliases and rename DataFrame columns based on aliases
    field_to_alias = {field_name: field.alias for field_name, field in model.__annotations__.items() if hasattr(field, 'alias')}
    df = df.rename(columns={v: k for k, v in field_to_alias.items()})

    # Validate and collect errors
    validated_rows, error_messages = [], []
    for idx, row in df.iterrows():
        try:
            instance = parse_obj_as(model, row.to_dict())
            validated_rows.append(instance.dict())
        except ValidationError as e:
            e = re.sub(r'For further information visit https://errors.pydantic.dev/\d+\.\d+/v/[a-z_]+', '', str(e))
            error_messages.append((idx, e))
            validated_rows.append(row.to_dict())  # Append the original row if validation fails
    
    # Print errors if requested
    if print_errors and error_messages:
        for idx, error in error_messages:
            print(f"Row {idx}: {error}")

    # Create a new DataFrame for validated rows
    validated_df = pd.DataFrame(validated_rows)
    valid_columns = [col for col in model.__annotations__.keys() if col in validated_df.columns]
    
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
