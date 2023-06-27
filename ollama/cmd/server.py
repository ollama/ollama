from aiohttp import web
import aiohttp_cors

from ollama import engine


def set_parser(parser):
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7734)
    parser.set_defaults(fn=serve)


def serve(models_home=".", *args, **kwargs):
    app = web.Application()

    cors = aiohttp_cors.setup(
        app,
        defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )

    app.add_routes(
        [
            web.post("/load", load),
            web.post("/unload", unload),
            web.post("/generate", generate),
        ]
    )

    for route in app.router.routes():
        cors.add(route)

    app.update(
        {
            "llms": {},
            "models_home": models_home,
        }
    )

    web.run_app(app, **kwargs)


async def load(request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise web.HTTPBadRequest()

    kwargs = {
        "llms": request.app.get("llms"),
        "models_home": request.app.get("models_home"),
    }

    engine.load(model, **kwargs)
    return web.Response()


async def unload(request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise web.HTTPBadRequest()

    engine.unload(model, llms=request.app.get("llms"))
    return web.Response()


async def generate(request):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise web.HTTPBadRequest()

    prompt = body.get("prompt")
    if not prompt:
        raise web.HTTPBadRequest()

    response = web.StreamResponse()
    await response.prepare(request)

    kwargs = {
        "llms": request.app.get("llms"),
        "models_home": request.app.get("models_home"),
    }

    for output in engine.generate(model, prompt, **kwargs):
        await response.write(output.encode("utf-8"))
        await response.write(b"\n")

    return response
