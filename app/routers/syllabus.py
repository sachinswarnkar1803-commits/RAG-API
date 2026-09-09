import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from app.models.schemas import TopicResponse, UnitResponse

router = APIRouter(prefix="/syllabus", tags=["Syllabus"])

def load_syllabus():
    with Path("data/syllabus.json").open("r", encoding="utf-8") as file:
        return json.load(file)

@router.get("", response_model=list[UnitResponse])
def get_syllabus():
    return load_syllabus()

@router.get("/units", response_model=list[UnitResponse])
def get_units():
    return load_syllabus()

@router.get("/units/{unit_id}")
def get_unit(unit_id: int):
    for unit in load_syllabus():
        if unit["unit_id"] == unit_id:
            return unit
    raise HTTPException(status_code=404, detail="Unit not found")

@router.get("/topics", response_model=list[TopicResponse])
def get_topics(unit_id: int | None = Query(default=None, ge=1, le=4)):
    topics = []
    for unit in load_syllabus():
        if unit_id is not None and unit["unit_id"] != unit_id:
            continue
        for topic in unit["topics"]:
            topics.append({"topic_id": topic["topic_id"], "unit_id": unit["unit_id"], "unit_title": unit["title"], "title": topic["title"], "description": topic["description"]})
    return topics

@router.get("/topics/{topic_id}", response_model=TopicResponse)
def get_topic(topic_id: str):
    for unit in load_syllabus():
        for topic in unit["topics"]:
            if topic["topic_id"] == topic_id:
                return {"topic_id": topic["topic_id"], "unit_id": unit["unit_id"], "unit_title": unit["title"], "title": topic["title"], "description": topic["description"]}
    raise HTTPException(status_code=404, detail="Topic not found")

@router.get("/raw-json")
def raw_json():
    return JSONResponse(content=load_syllabus())

@router.get("/html", response_class=HTMLResponse)
def syllabus_html():
    html = "<html><head><title>FastAPI Syllabus</title></head><body><h1>FastAPI Education Assistant</h1>"
    for unit in load_syllabus():
        html += f"<h2>Unit {unit['unit_id']}: {unit['title']}</h2><ul>"
        for topic in unit["topics"]:
            html += f"<li><b>{topic['topic_id']} - {topic['title']}</b>: {topic['description']}</li>"
        html += "</ul>"
    return HTMLResponse(content=html + "</body></html>")
