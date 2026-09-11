import os
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator
from jose import jwt, JWTError
import pypdf
from ml_model import NotesMLModel, train_from_notes_py

from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
NOTES_FILE = BASE_DIR / "data" / "notes.txt"
NOTES_PY_FILE = BASE_DIR / "notes.py"
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
SECRET_KEY = os.getenv("SECRET_KEY", "education-assistant-demo-secret")
ALGORITHM = "HS256"

# Active Document Tracker for Document Reader UI
active_document_info = {
    "title": "Default Notes (notes.py)",
    "filename": "notes.py",
    "text": "",
    "source_type": "default"
}

class Question(BaseModel):
    question: str
    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if len(value.strip()) < 2: raise ValueError("Question must contain at least 2 characters")
        return value.strip()

class TextInput(BaseModel):
    text: str
    title: str = "User Pasted Document"
    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if len(value.strip()) < 10: raise ValueError("Document text must contain at least 10 characters")
        return value.strip()

class AskResponse(BaseModel):
    found: bool
    answer: str
    source: str | None = None
    unit_or_section: int | None = None
    score: float
    results: list = []

class TrainingResponse(BaseModel):
    message: str
    documents: int
    vocabulary_size: int
    temporary: bool
    title: str = ""

class UserRegister(BaseModel):
    username: str
    password: str
    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        v = v.strip()
        if len(v) < 3: raise ValueError("Username must contain at least 3 characters")
        return v
    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        v = v.strip()
        if len(v) < 4: raise ValueError("Password must contain at least 4 characters")
        return v

class UserLoginJSON(BaseModel):
    username: str
    password: str

users = {"admin": "admin123", "student": "student123"}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(username: str):
    return jwt.encode({"sub": username}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users:
            raise HTTPException(401, "Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

ml_model = NotesMLModel()

def extract_text_from_file(filename: str, content_bytes: bytes) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages_text = []
            for page in reader.pages:
                t = page.extract_text()
                if t: pages_text.append(t)
            extracted = "\n\n".join(pages_text).strip()
            if not extracted:
                raise ValueError("Could not extract readable text from PDF pages.")
            return extracted
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF document: {str(e)}")
    else:
        try:
            return content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return content_bytes.decode("latin-1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_document_info
    if NOTES_PY_FILE.exists():
        _, res = train_from_notes_py(ml_model)
        from notes import NOTES_TEXT
        active_document_info = {
            "title": "Default Syllabus Notes (notes.py)",
            "filename": "notes.py",
            "text": NOTES_TEXT,
            "source_type": "default"
        }
    elif NOTES_FILE.exists():
        txt = NOTES_FILE.read_text(encoding="utf-8")
        ml_model.train(txt)
        active_document_info = {
            "title": "Default Syllabus Notes (notes.txt)",
            "filename": "notes.txt",
            "text": txt,
            "source_type": "default"
        }
    yield

app = FastAPI(
    title="FastAPI RAG Assistant & Document Reader API",
    description="Streamlined FastAPI backend with document reader, multi-format PDF/TXT/MD RAG training, and OAuth2/JWT authentication.",
    version="3.5",
    lifespan=lifespan
)

@app.get("/", response_class=HTMLResponse)
def home():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FastAPI RAG API Backend Running</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")

@app.get("/login-page", response_class=HTMLResponse)
@app.get("/login.html", response_class=HTMLResponse)
def login_page():
    login_path = STATIC_DIR / "login.html"
    if login_path.exists():
        return HTMLResponse(content=login_path.read_text(encoding="utf-8"))
    raise HTTPException(404, "login.html not found")

@app.post("/register", status_code=201)
def register(user_data: UserRegister):
    username = user_data.username.lower()
    if username in users:
        raise HTTPException(400, "Username already registered")
    users[username] = user_data.password
    token = create_access_token(username)
    return {
        "message": "User registered successfully",
        "username": username,
        "access_token": token,
        "token_type": "bearer"
    }

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username.lower()
    if username not in users or users[username] != form_data.password:
        raise HTTPException(401, "Incorrect username or password")
    return {"access_token": create_access_token(username), "token_type": "bearer"}

@app.post("/login/json")
def login_json(credentials: UserLoginJSON):
    username = credentials.username.lower()
    if username not in users or users[username] != credentials.password:
        raise HTTPException(401, "Incorrect username or password")
    return {"access_token": create_access_token(username), "token_type": "bearer", "username": username}

@app.post("/ask", response_model=AskResponse)
def ask(question: Question, user: str = Depends(get_current_user)):
    result = ml_model.answer(question.question)
    return {
        "found": result["found"],
        "answer": result["answer"],
        "source": result["source"],
        "unit_or_section": result.get("unit_or_section"),
        "score": result["score"],
        "results": [{"score": r["score"], "title": r["document"]["title"], "content": r["document"]["content"]} for r in result.get("results", [])]
    }

@app.post("/train/temporary", response_model=TrainingResponse)
async def train_temporary(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    allowed_exts = [".txt", ".pdf", ".md", ".doc", ".docx"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file type. Please upload one of: {', '.join(allowed_exts)}")

    content_bytes = await file.read()
    text = extract_text_from_file(file.filename, content_bytes)

    if not text.strip():
        raise HTTPException(400, "The uploaded document contains no readable text.")

    result = ml_model.train(text)

    global active_document_info
    active_document_info = {
        "title": f"Uploaded Document: {file.filename}",
        "filename": file.filename,
        "text": text,
        "source_type": "uploaded_file"
    }

    return {
        "message": f"Model successfully read and trained on document '{file.filename}'",
        "documents": result["documents"],
        "vocabulary_size": result["vocabulary_size"],
        "temporary": True,
        "title": file.filename
    }

@app.post("/train/text", response_model=TrainingResponse)
def train_user_text(input_data: TextInput, user: str = Depends(get_current_user)):
    text = input_data.text.strip()
    result = ml_model.train(text)

    global active_document_info
    active_document_info = {
        "title": input_data.title,
        "filename": "User Input Text",
        "text": text,
        "source_type": "user_text"
    }

    return {
        "message": f"Model trained successfully on user input text '{input_data.title}'",
        "documents": result["documents"],
        "vocabulary_size": result["vocabulary_size"],
        "temporary": True,
        "title": input_data.title
    }

@app.post("/train/notes-py", response_model=TrainingResponse)
def train_notes_py_route(user: str = Depends(get_current_user)):
    try:
        _, result = train_from_notes_py(ml_model)
        from notes import NOTES_TEXT
        global active_document_info
        active_document_info = {
            "title": "Default Syllabus Notes (notes.py)",
            "filename": "notes.py",
            "text": NOTES_TEXT,
            "source_type": "default"
        }
        return {
            "message": "Model trained successfully directly from notes.py module",
            "documents": result["documents"],
            "vocabulary_size": result["vocabulary_size"],
            "temporary": False,
            "title": "notes.py"
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to train from notes.py: {str(e)}")

@app.post("/model/reset")
def reset_model(user: str = Depends(get_current_user)):
    global active_document_info
    if NOTES_PY_FILE.exists():
        _, result = train_from_notes_py(ml_model)
        from notes import NOTES_TEXT
        active_document_info = {
            "title": "Default Syllabus Notes (notes.py)",
            "filename": "notes.py",
            "text": NOTES_TEXT,
            "source_type": "default"
        }
        return {"message": "Knowledge base reset to default notes.py", "documents": result["documents"], "vocabulary_size": result["vocabulary_size"]}
    elif NOTES_FILE.exists():
        txt = NOTES_FILE.read_text(encoding="utf-8")
        result = ml_model.train(txt)
        active_document_info = {
            "title": "Default Syllabus Notes (notes.txt)",
            "filename": "notes.txt",
            "text": txt,
            "source_type": "default"
        }
        return {"message": "Knowledge base reset to default notes.txt", "documents": result["documents"], "vocabulary_size": result["vocabulary_size"]}
    raise HTTPException(404, "Default notes file not found")

@app.get("/model/status")
def model_status(user: str = Depends(get_current_user)):
    return {
        "trained": ml_model.trained,
        "documents": len(ml_model.documents),
        "vocabulary_size": ml_model.vocabulary_size,
        "active_document": active_document_info["title"],
        "type": "TF-IDF + Cosine Similarity NLP Retrieval"
    }

@app.get("/model/reader")
def get_active_document(user: str = Depends(get_current_user)):
    return {
        "title": active_document_info["title"],
        "filename": active_document_info["filename"],
        "text": active_document_info["text"],
        "char_count": len(active_document_info["text"]),
        "sections": len(ml_model.documents),
        "vocabulary_size": ml_model.vocabulary_size,
        "source_type": active_document_info["source_type"]
    }

@app.get("/profile")
def profile(user: str = Depends(get_current_user)):
    return {"message": "Protected route accessed successfully", "current_user": user}
