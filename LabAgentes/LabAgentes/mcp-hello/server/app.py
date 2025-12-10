from fastapi import FastAPI, Request
from fastmcp import FastMCP

mcp = FastMCP(name="hello-mcp")

# ferramenta de exemplo
@mcp.tool()
def ping() -> str:
    "Retorna 'pong' para checagem rápida."
    return "pong"

# app ASGI do MCP via HTTP (usa SSE por baixo)
mcp_app = mcp.http_app()

# suas rotas opcionais (health, etc.)
app = FastAPI(title="Hello MCP App")

@app.get("/health")
def health():
    return {"status": "ok"}

# combine rotas MCP + suas, compartilhando lifespan do MCP
combined_app = FastAPI(
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)

# opcional: middleware (p.ex., capturar cabeçalhos)
@combined_app.middleware("http")
async def capture_headers(request: Request, call_next):
    return await call_next(request)