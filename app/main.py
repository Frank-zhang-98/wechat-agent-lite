from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api import router as api_router
from app.db import Base, engine, get_session
from app.services.model_pricing_service import warm_pricing_catalog
from app.services.scheduler_service import SchedulerService
from app.services.settings_service import SettingsService
from app import state

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
scheduler_service = SchedulerService()
state.scheduler = scheduler_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    with get_session() as session:
        SettingsService(session).ensure_defaults()
    try:
        warm_pricing_catalog()
    except Exception:
        pass
    scheduler_service.start()
    try:
        yield
    finally:
        scheduler_service.stop()


app = FastAPI(title="wechat-agent-lite", lifespan=lifespan)
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
