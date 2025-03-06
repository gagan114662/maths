"""
Web interface module for the trading system.
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Trading System Interface",
    description="Web interface for the Enhanced Trading Strategy System",
    version="1.0.0"
)

# Setup static files
static_path = Path(__file__).parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup templates
templates_path = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# Import routes after app initialization to avoid circular imports
from .app import WebInterface  # noqa: E402

# Create default interface instance
interface = WebInterface()

__all__ = ['app', 'interface', 'WebInterface']