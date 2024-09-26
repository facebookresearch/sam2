from aiohttp import web
from pydantic import ValidationError

import services
from models.requests import SAMRequest


async def segment(request: web.Request):
    try:
        data = await request.json()
        req = SAMRequest(**data)
        resp = await services.segment_image(req)
        return web.json_response({"message": resp}, status=200)
    except ValidationError as e:
        return web.json_response({"error": "invalid request"}, status=400)
