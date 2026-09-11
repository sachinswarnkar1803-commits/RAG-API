"""
Enhanced NLP + ML retrieval model for the FastAPI Education Assistant & RAG System.

This model uses TF-IDF-style weighting, section-aware note splitting,
and cosine similarity. It trains on notes.txt and retrieves relevant
sections from both FastAPI and NLP syllabus knowledge bases.
"""

import math
import re
from pathlib import Path


STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "to", "of", "in", "on", "for",
    "with", "as", "by", "from", "this", "that", "it", "be", "can", "used",
    "use", "using", "when", "what", "how", "why", "into", "than", "then",
    "they", "their", "its", "has", "have", "will", "may", "such", "also",
    "commonly", "normally", "example", "purpose", "api", "does", "do", "did"
}


def tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


INTENT_STOPWORDS = {
    "what", "is", "are", "a", "an", "the", "working", "work", "of", "how", "does", "do",
    "why", "use", "using", "difference", "between", "versus", "vs", "compare",
    "explain", "define", "definition", "meaning", "for", "in", "and", "or", "to",
    "with", "can", "used"
}


def simple_stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    return w


def extract_subject_tokens(question: str):
    tokens = tokenize(question)
    subject_tokens = [simple_stem(t) for t in tokens if t.lower() not in INTENT_STOPWORDS]
    if not subject_tokens:
        subject_tokens = [simple_stem(t) for t in tokens]
    return set(subject_tokens)


def split_notes(text: str):
    lines = text.splitlines()
    docs = []
    doc_id = 1
    current_title = "Overview & Fundamentals"
    current_lines = []

    for line in lines:
        line_s = line.strip()
        if not line_s:
            current_lines.append(line)
            continue

        is_numbered = re.match(r"^(\d+)\.\s+([A-Z0-9\s\-\–\—\(\)\/\,\:\?\!\.\=]+)$", line_s)
        is_unit = re.match(r"^UNIT\s+[IVXLCDM]+\s*[\—\–\-]\s*(.+)$", line_s, re.I)
        is_heading = (re.match(r"^[A-Z0-9\s\-\–\—\(\)\/\,]{4,65}$", line_s)
                      and len(line_s) > 4
                      and not line_s.startswith("HTTP")
                      and "..." not in line_s
                      and not line_s.startswith("===")
                      and not line_s.startswith("---"))

        if is_numbered or is_unit or (is_heading and len(current_lines) > 2):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if len(content) > 15:
                    docs.append({
                        "id": doc_id,
                        "title": current_title,
                        "content": content
                    })
                    doc_id += 1
            if is_numbered:
                current_title = f"{is_numbered.group(1)}. {is_numbered.group(2).title()}"
            elif is_unit:
                current_title = line_s.strip()
            else:
                current_title = line_s.title()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if len(content) > 15:
            docs.append({
                "id": doc_id,
                "title": current_title,
                "content": content
            })

    # Fallback for documents without structured headings (e.g. raw PDF text or user text input)
    if len(docs) <= 1:
        fallback_docs = []
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 15]
        if not paragraphs:
            words = text.split()
            chunk_size = 50
            for i in range(0, len(words), chunk_size):
                chunk_str = " ".join(words[i:i+chunk_size])
                if len(chunk_str) > 15:
                    fallback_docs.append({
                        "id": len(fallback_docs) + 1,
                        "title": f"Document Section {len(fallback_docs) + 1}",
                        "content": chunk_str
                    })
        else:
            for idx, p in enumerate(paragraphs, 1):
                p_lines = p.splitlines()
                title = p_lines[0].strip() if len(p_lines[0].strip()) < 50 else f"Section {idx}"
                fallback_docs.append({
                    "id": idx,
                    "title": title,
                    "content": p
                })
        if fallback_docs:
            return fallback_docs

    return docs


def synthesize_generative_answer(question: str, top_chunks: list) -> str:
    if not top_chunks:
        return "I could not find relevant information in the loaded document to answer this question."

    q_lower = question.lower().strip(" ?.")

    # Intent detection flags
    is_what = any(w in q_lower for w in ["what", "define", "meaning", "definition", "overview", "concept"])
    is_working = any(w in q_lower for w in ["working", "work", "how does", "how do", "process", "mechanism", "operates", "executes", "flow", "steps", "action"])
    is_why = any(w in q_lower for w in ["why", "benefit", "advantage", "purpose", "reason", "need", "why use"])
    is_diff = any(w in q_lower for w in ["difference", "diff", "versus", "vs", "compare", "distinguish", "contrasted"])
    is_example = any(w in q_lower for w in ["example", "sample", "syntax", "code", "how to write"])

    subject_tokens = extract_subject_tokens(question)

    raw_sentences = []
    for chunk in top_chunks:
        doc = chunk["document"]
        content = doc.get("content", "")
        # Strip header markers / bullet tags
        clean_content = re.sub(r"^[Q|\d\.\-]+:\s*", "", content, flags=re.MULTILINE)
        lines = clean_content.splitlines()
        for line in lines:
            line_s = line.strip()
            if not line_s or line_s.startswith("===") or line_s.startswith("---") or line_s.startswith("UNIT"):
                continue
            parts = re.split(r"(?<=[.!?])\s+|[\u2022\*\-\>]|\s*;\s*", line_s)
            for p in parts:
                p_clean = p.strip(" -*•>:\t")
                if len(p_clean) > 15:
                    raw_sentences.append(p_clean)

    scored_sentences = []
    seen = set()

    for idx, s in enumerate(raw_sentences):
        s_norm = s.lower()
        if s_norm in seen:
            continue
        seen.add(s_norm)

        s_tokens = set(simple_stem(t) for t in tokenize(s))
        overlap = len(subject_tokens.intersection(s_tokens))

        title_tokens = set(simple_stem(t) for t in tokenize(top_chunks[0]["document"].get("title", "")))
        title_overlap = len(subject_tokens.intersection(title_tokens))

        if len(subject_tokens) > 0 and overlap == 0 and title_overlap == 0:
            continue

        score = (overlap * 4.0) + (title_overlap * 1.5)

        if is_what:
            if any(kw in s_norm for kw in ["is a", "is used", "refers to", "defined as", "stands for", "is an", "class derived from", "provides", "represents"]):
                score += 5.0
            if any(kw in s_norm for kw in ["validation error", "422", "bad request", "does not match", "check required fields"]):
                score -= 4.0

        if is_working:
            if any(kw in s_norm for kw in ["inherit", "check", "validat", "coercion", "parse", "convert", "serialize", "raise", "custom", "enforce", "rule", "process", "execute", "run", "receive", "return", "automatically"]):
                score += 6.0

        if is_why:
            if any(kw in s_norm for kw in ["used to", "allows", "enables", "helps", "ensure", "prevent", "benefit", "purpose", "useful"]):
                score += 6.0
            if any(kw in s_norm for kw in ["validation error", "422", "bad request", "does not match", "check required fields"]):
                score -= 4.0

        if is_diff:
            if any(kw in s_norm for kw in ["difference", "while", "unlike", "whereas", "compared", "versus", "retrieves", "creates", "partially"]):
                score += 6.0

        if is_example:
            if any(kw in s_norm for kw in ["example", "such as", "for instance", "class ", "def ", "@app."]):
                score += 6.0

        if score > 0:
            scored_sentences.append((score, idx, s))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for sc, idx, s in scored_sentences:
        if len(selected) >= 3:
            break
        if not any(len(set(tokenize(s)).intersection(set(tokenize(sel)))) > 0.7 * min(len(tokenize(s)), len(tokenize(sel))) for sel in selected):
            selected.append(s)

    if not selected:
        fallback = top_chunks[0]["document"]["content"].strip()
        selected = [fallback[:250] + "..." if len(fallback) > 250 else fallback]

    subject_match = re.search(r"(?:what is|working of|why use|difference between|how does|explain|define)\s+([a-zA-Z0-9_\s]+)", q_lower)
    subject = subject_match.group(1).strip().title() if subject_match else ""
    if not subject and subject_tokens:
        subject = " ".join([t.capitalize() for t in subject_tokens])

    if is_working:
        prefix = f"Working & Operational Mechanism of {subject}:\n" if subject else "Working & Operational Process:\n"
    elif is_what:
        prefix = f"Definition & Overview of {subject}:\n" if subject else "Overview & Definition:\n"
    elif is_why:
        prefix = f"Purpose & Benefits of {subject}:\n" if subject else "Purpose & Key Benefits:\n"
    elif is_diff:
        prefix = f"Comparison Breakdown for {subject}:\n" if subject else "Comparative Analysis:\n"
    elif is_example:
        prefix = f"Examples & Usage of {subject}:\n" if subject else "Usage & Examples:\n"
    else:
        prefix = f"Synthesized Answer for '{question}':\n"

    body = " ".join(selected)
    if not body.endswith("."):
        body += "."

    return prefix + body


class NotesMLModel:
    def __init__(self):
        self.documents = []
        self.vectors = []
        self.idf = {}
        self.document_frequency = {}
        self.vocabulary_size = 0
        self.trained = False

    def train(self, notes_text: str):
        self.documents = split_notes(notes_text)

        token_lists = []
        for doc in self.documents:
            token_lists.append(tokenize(doc["title"] + "\n" + doc["content"]))

        self.document_frequency = {}
        for tokens in token_lists:
            for token in set(tokens):
                self.document_frequency[token] = (
                    self.document_frequency.get(token, 0) + 1
                )

        total = max(len(token_lists), 1)
        self.idf = {
            token: math.log((total + 1) / (1 + frequency)) + 1
            for token, frequency in self.document_frequency.items()
        }

        self.vocabulary_size = len(self.idf)
        self.vectors = [self._tfidf(tokens) for tokens in token_lists]
        self.trained = True

        return {
            "message": "Model trained successfully",
            "documents": len(self.documents),
            "vocabulary_size": self.vocabulary_size
        }

    def _tfidf(self, tokens):
        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        length = max(len(tokens), 1)
        return {
            token: (count / length) * self.idf.get(token, 0.0)
            for token, count in counts.items()
        }

    def _cosine_similarity(self, first, second):
        common = set(first).intersection(second)

        dot = sum(first[key] * second[key] for key in common)
        first_norm = math.sqrt(sum(value * value for value in first.values()))
        second_norm = math.sqrt(sum(value * value for value in second.values()))

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return dot / (first_norm * second_norm)

    def search(self, question: str, top_k: int = 3):
        if not self.trained:
            return []

        query_tokens = tokenize(question)
        query_vector = self._tfidf(query_tokens)

        results = []
        for index, vector in enumerate(self.vectors):
            doc = self.documents[index]
            title_lower = doc["title"].lower()
            if any(meta in title_lower for meta in ["out-of-scope", "unknown question", "paraphrased test inputs", "example user inputs", "nlp notes"]):
                continue

            score = self._cosine_similarity(query_vector, vector)
            if score > 0:
                results.append({
                    "score": round(score, 4),
                    "document": doc
                })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def answer(self, question: str):
        results = self.search(question, top_k=3)

        if not results or results[0]["score"] < 0.12:
            return {
                "found": False,
                "answer": "I could not find this topic in the provided notes.",
                "source": None,
                "unit_or_section": None,
                "score": 0.0,
                "results": []
            }

        best = results[0]
        generated_answer = synthesize_generative_answer(question, results)

        return {
            "found": True,
            "answer": generated_answer,
            "source": best["document"]["title"],
            "unit_or_section": best["document"]["id"],
            "score": best["score"],
            "results": results
        }


def train_from_notes_file(notes_file="data/notes.txt"):
    path = Path(notes_file)
    if not path.exists():
        raise FileNotFoundError(f"Notes file not found: {notes_file}")

    model = NotesMLModel()
    result = model.train(path.read_text(encoding="utf-8"))
    return model, result


def train_from_notes_py(model=None):
    """
    Trains the NotesMLModel directly using notes.py module.
    """
    try:
        from notes import NOTES_TEXT
    except ImportError:
        path = Path(__file__).resolve().parent / "notes.py"
        if not path.exists():
            raise FileNotFoundError("notes.py file not found")
        import importlib.util
        spec = importlib.util.spec_from_file_location("notes", str(path))
        notes_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(notes_mod)
        NOTES_TEXT = notes_mod.NOTES_TEXT

    if model is None:
        model = NotesMLModel()
    result = model.train(NOTES_TEXT)
    return model, result

