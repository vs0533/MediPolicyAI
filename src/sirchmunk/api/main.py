# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Main FastAPI application for Sirchmunk API
Combines all API modules and provides centralized configuration.

When the SIRCHMUNK_SERVE_UI environment variable is set to "true",
the application also serves the pre-built WebUI static files from
the Sirchmunk cache directory, enabling single-port access to both
the API and the WebUI.
"""

import os
from pathlib import Path

# Load .env file from Sirchmunk work directory before any module imports.
# This ensures environment variables (LLM_API_KEY, LLM_BASE_URL, etc.) are
# available when constants.py and other modules are first imported.
_work_path = Path(
    os.getenv("SIRCHMUNK_WORK_PATH", os.path.expanduser("~/.sirchmunk"))
).expanduser().resolve()
_env_file = _work_path / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_env_file), override=False)
    except ImportError:
        # Fallback: manual .env parsing if python-dotenv is not installed
        try:
            with open(_env_file, "r") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _key, _, _val = _line.partition("=")
                        _key = _key.strip()
                        _val = _val.strip().strip('"').strip("'")
                        if _key and _key not in os.environ:
                            os.environ[_key] = _val
        except Exception:
            pass

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse, Response

from .security import SecurityHeadersMiddleware, is_public_service_mode, verify_token

logger = logging.getLogger(__name__)

# Import all API routers
from .knowledge import router as knowledge_router
from .settings import router as settings_router
from .history import router as history_router, dashboard_router
from .chat import router as chat_router
from .monitor import router as monitor_router
from .search import router as search_router
from .files import router as files_router

# Determine whether to serve the WebUI static files.
# Set by `sirchmunk web serve` via environment variable.
_serve_ui = os.getenv("SIRCHMUNK_SERVE_UI") == "true"
_static_dir = _work_path / ".cache" / "web_static"
_ui_available = _serve_ui and _static_dir.is_dir() and (_static_dir / "index.html").exists()

# Create FastAPI application
_debug = os.getenv("SIRCHMUNK_DEBUG", "false").lower() == "true"

app = FastAPI(
    title="政策问答 API",
    description="医保政策公共问答服务 API",
    version="1.0.0",
    docs_url="/docs" if _debug else None,
    redoc_url="/redoc" if _debug else None,
)

# Configure CORS
_allowed_origins_raw = os.getenv("SIRCHMUNK_ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
if not _allowed_origins:
    _allowed_origins = ["*"]  # backward-compatible when unconfigured

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce Bearer-token auth on /api/ routes when SIRCHMUNK_API_TOKEN is set."""
    path = request.url.path
    if is_public_service_mode() and path.startswith("/api/"):
        public_allowed = path == "/api/v1/settings/ui"
        blocked_prefixes = (
            "/api/v1/settings",
            "/api/v1/monitor",
            "/api/v1/knowledge",
            "/api/v1/files",
            "/api/v1/file-picker",
            "/api/v1/file-browser",
            "/api/v1/search",
            "/api/v1/history",
            "/api/v1/dashboard",
            "/api/v1/chat/sessions",
        )
        if not public_allowed and any(path.startswith(prefix) for prefix in blocked_prefixes):
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "error": {
                        "code": "PUBLIC_SERVICE_FORBIDDEN",
                        "message": "This public service endpoint is not available.",
                    },
                },
            )

    # Exempt: health check, favicon, static assets, OPTIONS preflight, UI settings
    exempt = (
        path in ("/health", "/favicon.ico")
        or path.startswith("/_next/")
        or path == "/api/v1/settings/ui"
        or request.method == "OPTIONS"
    )
    if not exempt and path.startswith("/api/"):
        await verify_token(request)
    response = await call_next(request)
    return response


@app.on_event("startup")
def _prewarm_chat_search():
    """Create the chat search singleton at startup so the embedding model starts loading immediately.
    This reduces the chance of the first user request blocking on model load (e.g. in Docker).
    """
    try:
        from .chat import get_search_instance
        get_search_instance()
    except Exception:
        pass


# Include all API routers (registered before static mount so they take priority)
app.include_router(knowledge_router)
app.include_router(settings_router)
app.include_router(history_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(monitor_router)
app.include_router(search_router)
app.include_router(files_router)

# Root endpoint: return API info when UI is not served,
# otherwise let the static mount handle "/"
if not _ui_available:
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "政策问答 API",
            "version": "1.0.0",
            "description": "医保政策公共问答服务 API",
            "status": "running",
            "endpoints": {
                "search": "/api/v1/search",
                "knowledge": "/api/v1/knowledge",
                "settings": "/api/v1/settings",
                "history": "/api/v1/history",
                "chat": "/api/v1/chat",
                "monitor": "/api/v1/monitor"
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ui_enabled": _ui_available,
        "services": {
            "api": "running",
            "database": "connected",
            "llm": "available",
            "embedding": "available"
        }
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred",
                "details": "Please try again later or contact support"
            }
        }
    )

# Favicon route — browsers always request /favicon.ico; serve it from the
# static directory when available, otherwise return 204 No Content to avoid 404.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if _ui_available:
        ico_path = _static_dir / "favicon.ico"
        if ico_path.is_file():
            return FileResponse(ico_path, media_type="image/x-icon")
    return Response(status_code=204)


# Mount static files for WebUI when enabled.
# This MUST be after all API route registrations so that API endpoints
# take priority over the catch-all static file serving.
if _ui_available:
    from fastapi.staticfiles import StaticFiles

    # SPA route fallback — Next.js static export creates route directories
    # (e.g. history/) for RSC payloads but no index.html inside them.
    # Starlette's StaticFiles(html=True) resolves the directory first and
    # returns 404 when index.html is missing. This middleware intercepts
    # those 404s for known frontend routes and serves the correct .html file.
    _SPA_ROUTES = {"history", "knowledge", "monitor", "settings"}

    @app.middleware("http")
    async def spa_fallback(request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 404:
            path = request.url.path.strip("/").split("/")[0]
            if path in _SPA_ROUTES:
                html_file = _static_dir / f"{path}.html"
                if html_file.is_file():
                    if request.method == "HEAD":
                        return Response(
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                        )
                    return FileResponse(html_file, media_type="text/html")
        return response

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="ui")
    logger.info("WebUI enabled, serving static files from %s", _static_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8584,
        reload=True,
        log_level="info"
    )
