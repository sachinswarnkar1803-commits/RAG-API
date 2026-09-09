# FastAPI Education Assistant — RAG API

**Backend-only project. No frontend application is included.**

A practical FastAPI RAG project based on the syllabus through OAuth2/JWT and written in a simple, classroom-friendly style inspired by the teacher reference practical.

## Concepts covered

### Unit I — REST & FastAPI Fundamentals
- HTTP verbs, status codes and headers
- FastAPI project setup and endpoints
- request/query/path/body parameters
- response models and status codes
- Swagger `/docs` and ReDoc `/redoc`

### Unit II — Data Validation & Error Handling
- Pydantic request validation
- nested models and type coercion
- Enum and custom `field_validator`
- automatic serialization
- HTTPException and custom error handler
- JSONResponse, HTMLResponse and FileResponse
- UploadFile/File upload and download

### Unit III — Async Programming & Dependency Injection
- sync and async routes
- async endpoint/coroutine
- BackgroundTasks
- dependency injection and `Depends()`
- pydantic-settings environment configuration
- modular project structure

### Unit IV — OAuth2/JWT
- OAuth2 password bearer
- JWT access tokens
- protected RAG routes
- Swagger Authorize flow

## RAG
The knowledge base is the FastAPI syllabus in `data/syllabus.json`. Retrieval uses TF-IDF plus cosine similarity. The answer is generated from the most relevant syllabus topics, keeping the project local and easy to explain in a practical/viva.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`.

## Demo credentials
`student / student123`
`admin / admin123`

## Suggested Swagger demonstration
1. GET `/`
2. GET `/syllabus`
3. GET `/syllabus/topics?unit_id=2`
4. POST `/profile/validate`
5. GET `/rag/search?query=pydantic`
6. POST `/auth/token`
7. Swagger Authorize
8. POST `/rag/ask`
9. POST `/files/upload`
10. GET `/files/download/{filename}`
11. GET `/syllabus/html`
12. GET `/rag/async-demo`


## Backend-only note

This project contains only the FastAPI backend, API routes, models, services,
dependencies, configuration, RAG logic, authentication, and data files.

There is no React, HTML frontend application, CSS, JavaScript frontend,
static website, or frontend build system.

`/syllabus/html` returns an `HTMLResponse` because HTML response classes are
explicitly required by the syllabus. It is a backend API response used to
demonstrate the FastAPI `HTMLResponse` concept; it is not a frontend project.

Swagger `/docs` and ReDoc `/redoc` are FastAPI's automatically generated API
documentation and testing interfaces, not custom frontend code.
