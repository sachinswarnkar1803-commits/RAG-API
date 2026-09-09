import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGService:
    def __init__(self, data_file: str = "data/syllabus.json"):
        self.data_file = Path(data_file)
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.load_documents()

    def load_documents(self):
        with self.data_file.open("r", encoding="utf-8") as file:
            units = json.load(file)
        self.documents = []
        for unit in units:
            for topic in unit["topics"]:
                content = (f"Unit {unit['unit_id']} {unit['title']}. "
                           f"Topic {topic['topic_id']} {topic['title']}. "
                           f"{topic['description']}")
                self.documents.append({
                    "topic_id": topic["topic_id"],
                    "unit_id": unit["unit_id"],
                    "unit_title": unit["title"],
                    "topic": topic["title"],
                    "content": content,
                })
        self.matrix = self.vectorizer.fit_transform([d["content"] for d in self.documents])

    def search(self, query: str, top_k: int = 3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        indexes = scores.argsort()[::-1][:top_k]
        results = []
        for index in indexes:
            score = float(scores[index])
            if score > 0:
                results.append({**self.documents[index], "score": round(score, 4)})
        return results

    def generate_answer(self, question: str, results: list[dict]):
        if not results:
            return ("I could not find a relevant topic in the FastAPI syllabus. "
                    "Try REST, Pydantic, async programming, dependency injection, "
                    "file handling, or OAuth2/JWT.")
        bullets = [f"- {r['topic']}: {r['content']}" for r in results]
        return f"Based on the syllabus, the answer to '{question}' is:\n" + "\n".join(bullets)


rag_service = RAGService()
