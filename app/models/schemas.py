from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TopicType(str, Enum):
    THEORY = "theory"
    PRACTICAL = "practical"
    SECURITY = "security"


class Address(BaseModel):
    city: str
    state: str
    pincode: str


class StudentProfile(BaseModel):
    name: str
    email: str
    topic_type: TopicType = TopicType.THEORY
    address: Address

    @field_validator("name")
    @classmethod
    def check_name(cls, value):
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Name must contain at least 2 characters")
        return value


class UnitResponse(BaseModel):
    unit_id: int
    title: str
    hours: int


class TopicResponse(BaseModel):
    topic_id: str
    unit_id: int
    unit_title: str
    title: str
    description: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)
    topic_type: Optional[TopicType] = None

    @field_validator("question")
    @classmethod
    def check_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value


class Source(BaseModel):
    topic_id: str
    unit_id: int
    topic: str
    score: float
    content: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    current_user: Optional[str] = None


class StudentResponse(BaseModel):
    id: int
    name: str
    email: str
    topic_type: TopicType
    address: Address
