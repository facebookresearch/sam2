from aiohttp import web


def health_check(request):
    return web.json_response({"message": "Success"})
