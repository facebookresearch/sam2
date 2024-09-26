from aiohttp import web
from views import segment, health_check

app = web.Application()

app.router.add_get("/", health_check)
app.router.add_post("/segment", segment)

web.run_app(app)
