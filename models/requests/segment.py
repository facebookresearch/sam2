from typing import List

from pydantic import BaseModel

from enums import MediaType, SAMModels


class Pointers(BaseModel):
    x: int
    y: int
    label: int


class SAMRequest(BaseModel):
    media_type: MediaType
    media_url: str
    model: SAMModels
    pointers: List[Pointers]
