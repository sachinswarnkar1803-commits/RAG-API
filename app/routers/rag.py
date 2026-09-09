from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from app.dependencies.dependencies import get_current_user, get_rag_service
from app.models.schemas import QuestionRequest, RAGResponse, Source
from app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG Assistant"])

def write_question_log(question: str, username: str, answer: str, log_file: str):
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"User: {username} | Question: {question} | Answer: {answer}\n")

@router.get("/search")
def search_syllabus(query: str = Query(..., min_length=2, max_length=200), top_k: int = Query(default=3, ge=1, le=5), rag: RAGService = Depends(get_rag_service)):
    return {"query": query, "results": rag.search(query, top_k)}

@router.post("/ask", response_model=RAGResponse)
def ask_question(request: QuestionRequest, background_tasks: BackgroundTasks, rag: RAGService = Depends(get_rag_service), current_user: str = Depends(get_current_user)):
    results = rag.search(request.question, request.top_k)
    answer = rag.generate_answer(request.question, results)
    sources = [Source(topic_id=r["topic_id"], unit_id=r["unit_id"], topic=r["topic"], score=r["score"], content=r["content"]) for r in results]
    background_tasks.add_task(write_question_log, request.question, current_user, answer, "logs/questions.txt")
    return {"question": request.question, "answer": answer, "sources": sources, "current_user": current_user}

@router.post("/ask-json")
def ask_question_json(request: QuestionRequest, rag: RAGService = Depends(get_rag_service)):
    results = rag.search(request.question, request.top_k)
    answer = rag.generate_answer(request.question, results)
    return JSONResponse(content={"question": request.question, "answer": answer, "source_count": len(results)})

@router.get("/async-demo")
async def async_demo():
    return {"message": "This is an asynchronous RAG endpoint.", "note": "FastAPI supports both sync and async path operations."}
