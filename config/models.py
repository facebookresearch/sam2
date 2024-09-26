from pydantic import BaseModel


class Paths(BaseModel):
    tmp_file_dir: str


class AmazonS3(BaseModel):
    bucket: str


class Storage(BaseModel):
    s3: AmazonS3


class Configuration(BaseModel):
    paths: Paths
    storage: Storage
