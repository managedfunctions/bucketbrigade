import arrow
import re
from pydantic import (
    BaseModel,
    computed_field,
)



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
    environment: str = "prod"
    timezone: str

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
    def archive_input_path(self) -> str:
        return f"{self.folder_path}/archive_input_folder"

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
        return arrow.now(self.timezone).isoformat("YYYYMMDDHHmmss")
