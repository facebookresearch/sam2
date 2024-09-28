# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union


class MultipartResponseBuilder:
    message: bytes

    def __init__(self, boundary: str) -> None:
        self.message = b"--" + boundary.encode("utf-8") + b"\r\n"

    @classmethod
    def build(
        cls, boundary: str, headers: Dict[str, str], body: Union[str, bytes]
    ) -> "MultipartResponseBuilder":
        builder = cls(boundary=boundary)
        for k, v in headers.items():
            builder.__append_header(key=k, value=v)
        if isinstance(body, bytes):
            builder.__append_body(body)
        elif isinstance(body, str):
            builder.__append_body(body.encode("utf-8"))
        else:
            raise ValueError(
                f"body needs to be of type bytes or str but got {type(body)}"
            )

        return builder

    def get_message(self) -> bytes:
        return self.message

    def __append_header(self, key: str, value: str) -> "MultipartResponseBuilder":
        self.message += key.encode("utf-8") + b": " + value.encode("utf-8") + b"\r\n"
        return self

    def __close_header(self) -> "MultipartResponseBuilder":
        self.message += b"\r\n"
        return self

    def __append_body(self, body: bytes) -> "MultipartResponseBuilder":
        self.__append_header(key="Content-Length", value=str(len(body)))
        self.__close_header()
        self.message += body
        return self
