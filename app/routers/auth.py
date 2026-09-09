from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from app.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])
users = {"student": {"username": "student", "password": "student123", "role": "student"}, "admin": {"username": "admin", "password": "admin123", "role": "admin"}}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def authenticate_user(username: str, password: str):
    user = users.get(username)
    if not user or user["password"] != password:
        return None
    return user

@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    token = jwt.encode({"sub": user["username"], "role": user["role"], "exp": expire}, settings.secret_key, algorithm="HS256")
    return {"access_token": token, "token_type": "bearer"}

@router.get("/demo-users")
def demo_users():
    return {"student": "student123", "admin": "admin123", "note": "Demo credentials only. Do not use real passwords."}
