from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.config import settings
from app.routers import auth, files, profile, rag, syllabus

app = FastAPI(title=settings.app_name, version=settings.app_version, description=(
    "A syllabus-based RAG API demonstrating REST, Pydantic, validation, "
    "response classes, files, async routes, BackgroundTasks, Depends(), "
    "settings and OAuth2/JWT."
))

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "InternalServerError", "message": "An unexpected error occurred.", "path": str(request.url.path)})

@app.get("/", tags=["Basic"])
def root():
    return {"message": "FastAPI Education Assistant - RAG API", "status": "running", "docs": "/docs", "redoc": "/redoc"}

@app.get("/health", tags=["Basic"])
def health():
    return {"status": "healthy", "service": "FastAPI Education Assistant"}

app.include_router(auth.router)
app.include_router(syllabus.router)
app.include_router(rag.router)
app.include_router(files.router)
app.include_router(profile.router)
