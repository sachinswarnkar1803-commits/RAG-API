from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from app.config import settings

router = APIRouter(prefix="/files", tags=["Files"])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    allowed_extensions = {".txt", ".md", ".json"}
    extension = Path(file.filename or "").suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only .txt, .md and .json files are allowed.")
    target = upload_dir / Path(file.filename).name
    content = await file.read()
    target.write_bytes(content)
    return {"message": "File uploaded successfully", "filename": target.name, "size": len(content)}

@router.get("/download/{filename}")
def download_file(filename: str):
    file_path = Path(settings.upload_dir) / Path(filename).name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="application/octet-stream")
