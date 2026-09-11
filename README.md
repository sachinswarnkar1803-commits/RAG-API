# FastAPI RAG Assistant & Document Reader API

A streamlined **Retrieval-Augmented Generation (RAG) API** built with **FastAPI**, featuring:
- 🔐 JWT OAuth2 Authentication (Login, Register)
- 📄 Multi-format Document Ingestion (PDF, TXT, MD)
- 🧠 TF-IDF + Cosine Similarity Retrieval Model (zero external ML dependencies)
- 💡 Intent-Aware Generative Answer Synthesis (what/working/why/difference/example)
- 🌐 Glassmorphism Web UI (Dashboard + Standalone Login Page)
- ✅ 6/6 Pytest tests passing

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)
```bash
cp .env.example .env
# Edit .env with your own SECRET_KEY
```

### 3. Run the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
- **Dashboard UI** → http://127.0.0.1:8000
- **Login Page** → http://127.0.0.1:8000/login-page
- **Swagger Docs** → http://127.0.0.1:8000/docs

---

## 🔑 Default Credentials

| Username | Password |
|----------|----------|
| `admin`  | `admin123` |
| `student`| `student123` |

---

## 📡 Main API Routes

### Interface
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Main Dashboard UI |
| GET | `/login-page` | Standalone Login & Register page |

### Authentication
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/register` | Register a new user → returns JWT token |
| POST | `/login` | OAuth2 form login |
| POST | `/login/json` | JSON body login (`{"username":"...","password":"..."}`) |
| GET | `/profile` | View authenticated user profile |

### RAG Q&A
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/ask` | Ask a question — returns intent-synthesized generative answer |

### Document Reader & Training
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/train/temporary` | Upload a PDF/TXT/MD file — model reads & trains on it |
| POST | `/train/text` | Train model on raw pasted text input |
| POST | `/train/notes-py` | Train model from embedded `notes.py` knowledge base |
| GET | `/model/reader` | Fetch active document text & training stats |
| GET | `/model/status` | Check model training status |
| POST | `/model/reset` | Reset model to default `notes.py` knowledge base |

---

## 🧪 Run Tests

```bash
pytest test_main.py -v
```

---

## 📁 Project Structure

```
RAG-API/
├── main.py              # FastAPI app & all routes
├── ml_model.py          # TF-IDF retrieval + generative answer synthesis
├── notes.py             # Embedded knowledge base (NLP + FastAPI notes)
├── test_main.py         # Pytest test suite
├── requirements.txt     # Python dependencies
├── data/
│   ├── notes.txt        # Plaintext knowledge base (fallback)
│   └── NLP_notes.txt    # NLP-specific notes
├── static/
│   ├── index.html       # Main dashboard UI
│   └── login.html       # Standalone login/register page
└── uploads/             # User-uploaded files (gitignored)
```

---

## 🛠 Tech Stack

- **FastAPI** — Web framework
- **Pydantic v2** — Request validation
- **python-jose** — JWT token generation & verification
- **pypdf** — PDF text extraction
- **Custom TF-IDF** — No external ML library required
- **Vanilla HTML/CSS/JS** — Frontend (glassmorphism design)
