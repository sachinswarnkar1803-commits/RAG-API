from fastapi import APIRouter, Depends
from app.dependencies.dependencies import get_current_user
from app.models.schemas import StudentProfile

router = APIRouter(prefix="/profile", tags=["Dependency Injection"])

@router.get("")
def profile(current_user: str = Depends(get_current_user)):
    return {"message": "Dependency Injection Example", "current_user": current_user}

@router.post("/validate")
def validate_profile(profile: StudentProfile):
    return {"message": "Nested Pydantic model validated successfully", "profile": profile}
