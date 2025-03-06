"""
Web interface for the trading system.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
import jwt

from ..agents import AgentFactory
from ..strategies.templates import StrategyFactory
from ..utils.config import load_config

logger = logging.getLogger(__name__)

# Get app instance from __init__.py
from . import app, templates

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load configuration
config = load_config()
SECRET_KEY = config.get('web', {}).get('secret_key', 'your-secret-key')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class WebInterface:
    """Web interface for trading system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize web interface."""
        self.config = load_config(config_path) if config_path else load_config()
        self.agent_factory = AgentFactory()
        self.strategy_factory = StrategyFactory()
        self.active_websockets: List[WebSocket] = []
        
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user."""
        user = await self._get_user(username)
        if not user or not self._verify_password(password, user['hashed_password']):
            return None
        return user
        
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return pwd_context.verify(plain_password, hashed_password)
        
    async def _get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user details."""
        # TODO: Implement user storage and retrieval
        # For now, return a test user
        if username == "test":
            return {
                "username": "test",
                "hashed_password": pwd_context.hash("test"),
                "is_active": True
            }
        return None
        
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
    async def get_current_user(self, token: str = Depends(oauth2_scheme)):
        """Get current user from token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if not username:
                raise HTTPException(status_code=401)
            user = await self._get_user(username)
            if not user:
                raise HTTPException(status_code=401)
            return user
        except jwt.JWTError:
            raise HTTPException(status_code=401)
            
    async def broadcast_update(self, update: Dict[str, Any]):
        """Broadcast update to all connected clients."""
        for websocket in self.active_websockets:
            try:
                await websocket.send_json(update)
            except Exception as e:
                logger.error(f"Error broadcasting update: {str(e)}")
                try:
                    self.active_websockets.remove(websocket)
                except ValueError:
                    pass

# Create global interface instance
interface = WebInterface()

# Route handlers

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint."""
    user = await interface.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    access_token = interface.create_access_token(
        data={"sub": user["username"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/strategies")
async def get_strategies(user: Dict = Depends(interface.get_current_user)):
    """Get all strategies."""
    try:
        strategies = []
        for agent in interface.agent_factory.get_agents_by_type('generation'):
            strategies.extend(await agent.get_generated_strategies())
        return strategies
    except Exception as e:
        logger.error(f"Error getting strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/strategies")
async def create_strategy(
    strategy_config: Dict[str, Any],
    user: Dict = Depends(interface.get_current_user)
):
    """Create new strategy."""
    try:
        strategy = interface.strategy_factory.create_strategy(strategy_config)
        generator = interface.agent_factory.create_agent('generation')
        await generator.generate_strategy(strategy_config)
        return {"status": "success", "strategy_id": strategy.id}
    except Exception as e:
        logger.error(f"Error creating strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    interface.active_websockets.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            # Handle different message types
            if data.get('type') == 'subscribe':
                await websocket.send_json({'status': 'subscribed'})
            elif data.get('type') == 'unsubscribe':
                await websocket.send_json({'status': 'unsubscribed'})
            else:
                await websocket.send_json({
                    'status': 'error',
                    'message': 'Unknown message type'
                })
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        try:
            interface.active_websockets.remove(websocket)
        except ValueError:
            pass