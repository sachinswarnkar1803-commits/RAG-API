"""
NLP & FastAPI Notes Knowledge Base Module for RAG Model Training.
This module defines NOTES_TEXT and helper methods to train NotesMLModel directly on notes.py.
"""

from pathlib import Path

NOTES_TEXT = r"""FASTAPI EDUCATION ASSISTANT — RAG API
DETAILED API DEVELOPMENT NOTES
================================

PURPOSE
-------
This file is a study/knowledge-base document for a FastAPI Education Assistant RAG API. It covers REST and FastAPI fundamentals, data validation, error handling, async programming, dependency injection, settings, project structure, OAuth2/JWT, and simple educational RAG retrieval.

============================================================
UNIT 1 — REST & FASTAPI FUNDAMENTALS
============================================================

1. HTTP AND REST RECAP
----------------------
HTTP (HyperText Transfer Protocol) is used for communication between clients and servers. A client sends an HTTP request and the server returns an HTTP response.

An HTTP request can contain a method, URL/path, headers, query parameters, and a body. A response contains a status code, headers, and a response body.

HTTP METHODS:
GET retrieves data. Example: GET /students
POST creates/submits data. Example: POST /students
PUT replaces/updates an existing resource. Example: PUT /students/10
PATCH partially updates an existing resource. Example: PATCH /students/10
DELETE removes a resource. Example: DELETE /students/10

IMPORTANT STATUS CODES:
200 OK — successful request.
201 Created — resource created successfully.
204 No Content — successful request with no response body.
400 Bad Request — invalid request.
401 Unauthorized — authentication is missing or invalid.
403 Forbidden — authenticated user lacks permission.
404 Not Found — requested resource does not exist.
422 Unprocessable Entity — request data failed validation; FastAPI commonly uses this for validation errors.
500 Internal Server Error — unexpected server-side error.

COMMON HEADERS:
Content-Type describes the request/response data format, for example application/json.
Authorization carries authentication information, commonly: Authorization: Bearer <token>.
Accept indicates response formats the client can accept.

REST (Representational State Transfer) is an architectural style for designing APIs around resources. A REST API normally uses nouns in URLs and HTTP methods for actions.
Good examples: GET /students, POST /students, GET /students/10, PUT /students/10, DELETE /students/10.
Avoid unnecessary action-style URLs such as /getStudents when the HTTP method already expresses the operation.

2. FASTAPI PROJECT SETUP
------------------------
FastAPI is a Python framework for building web APIs. Basic installation:
pip install fastapi uvicorn

A minimal application:
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello FastAPI"}

Run with:
uvicorn main:app --reload

Here main is the Python file, app is the FastAPI object, and --reload restarts the development server when code changes.

FastAPI automatically generates documentation at /docs (Swagger UI) and /redoc (ReDoc). These are backend documentation/testing interfaces and do not require a custom frontend.

3. CREATING API ENDPOINTS
-------------------------
An endpoint combines a URL/path and an HTTP method. FastAPI uses decorators such as @app.get(), @app.post(), @app.put(), @app.patch(), and @app.delete().

Example:
@app.get("/hello")
def hello():
    return {"message": "Hello"}

4. REQUEST AND RESPONSE OBJECTS
-------------------------------
A request contains information sent by the client. A response contains information returned by the API. FastAPI can parse JSON request bodies into Pydantic models.

Example:
class Student(BaseModel):
    name: str
    age: int

@app.post("/students")
def create_student(student: Student):
    return student

FastAPI can also expose the Request object when low-level request information is needed, for example request.method or request.headers.

5. PATH PARAMETERS
------------------
Path parameters are values embedded in a URL.

@app.get("/students/{student_id}")
def get_student(student_id: int):
    return {"student_id": student_id}

GET /students/5 gives student_id = 5. FastAPI validates the declared type.
Path parameters are useful for identifying a specific resource.

6. QUERY PARAMETERS
-------------------
Query parameters appear after ? in a URL.
Example: GET /students?course=DataScience

They are useful for searching, filtering, sorting, pagination, and optional settings.
Example:
@app.get("/search")
def search(query: str):
    return {"query": query}

7. REQUEST BODY PARAMETERS
--------------------------
A request body carries structured data, commonly JSON. Pydantic models define the expected body.

class StudentCreate(BaseModel):
    name: str
    age: int
    email: str

@app.post("/students")
def create_student(student: StudentCreate):
    return student

8. RESPONSE MODELS
------------------
A response model defines and validates the structure returned by an endpoint and improves Swagger documentation.

class StudentResponse(BaseModel):
    id: int
    name: str
    email: str

@app.get("/students/{student_id}", response_model=StudentResponse)
def get_student(student_id: int):
    return {"id": 1, "name": "Amit", "email": "amit@example.com"}

9. SWAGGER AND REDOC
--------------------
Swagger UI is normally available at /docs. It lets developers view endpoints, enter parameters, send request bodies, authorize, execute calls, and inspect responses.
ReDoc is normally available at /redoc and presents the OpenAPI documentation in another format.

============================================================
UNIT 2 — DATA VALIDATION & ERROR HANDLING
============================================================

10. PYDANTIC FOR REQUEST VALIDATION
-----------------------------------
Pydantic is used by FastAPI for data validation and parsing. Models inherit from BaseModel and define expected fields and types.

class Student(BaseModel):
    name: str
    age: int
    email: str

FastAPI automatically checks incoming data against the model and returns a validation error when required data is missing or incompatible.

11. NESTED MODELS
-----------------
A Pydantic model can contain another model.

class Address(BaseModel):
    city: str
    state: str
    pincode: str

class StudentCreate(BaseModel):
    name: str
    age: int
    address: Address

Example JSON:
{"name":"Ravi","age":21,"address":{"city":"Nashik","state":"Maharashtra","pincode":"422001"}}

Nested models are useful when related information has its own structure.

12. TYPE COERCION
-----------------
Pydantic can parse compatible input into declared Python types. For example, numeric input may be converted to int/float when allowed by validation rules. Invalid values cannot be converted and cause validation errors. Clients should still send correct data types.

13. ENUMS
---------
Enum restricts a field to a fixed set of values.

from enum import Enum
class Course(str, Enum):
    AI = "AI"
    DATASCIENCE = "DataScience"
    COMPUTER_SCIENCE = "ComputerScience"

A model using course: Course only accepts the defined choices. Enums are useful for course, category, difficulty, status, and role fields.

14. CUSTOM VALIDATION
---------------------
Custom validators enforce rules beyond basic types.

from pydantic import BaseModel, field_validator
class Student(BaseModel):
    name: str
    age: int
    @field_validator("age")
    @classmethod
    def validate_age(cls, value):
        if value < 18 or value > 60:
            raise ValueError("Age must be between 18 and 60")
        return value

A validator raises an error for invalid values and returns valid values.

15. AUTOMATIC DATA SERIALIZATION
--------------------------------
Serialization converts Python/model data into a format such as JSON for the client. FastAPI and Pydantic handle much of this automatically.

16. HTTPException
-----------------
HTTPException is used to return a specific HTTP error.

Example:
raise HTTPException(status_code=404, detail="Student not found")

Common uses include missing resources, invalid operations, authentication failures, and permission errors.

17. CUSTOM ERROR RESPONSES
---------------------------
Error responses should be clear and useful. Example:
raise HTTPException(status_code=400, detail="Invalid course selected")

18. RESPONSE CLASSES
--------------------
JSONResponse explicitly returns JSON.
HTMLResponse returns HTML content.
FileResponse returns a file to the client.

Examples:
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

JSONResponse(content={"message":"JSON response"})
HTMLResponse("<h1>FastAPI</h1>")
FileResponse("notes.txt")

HTMLResponse is still a backend response class; it does not mean a separate frontend must be created.

19. FILE UPLOADS
----------------
FastAPI can receive uploaded files using UploadFile and File.

Example:
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}

UploadFile provides filename, content type, and a file object. File uploads commonly require python-multipart.

20. FILE DOWNLOADS
------------------
FileResponse can return a stored file.

@app.get("/download")
def download():
    return FileResponse("notes.txt")

Always check that a requested file exists and safely handle file paths.

============================================================
UNIT 3 — ASYNC PROGRAMMING & DEPENDENCY INJECTION
============================================================

21. SYNC VS ASYNC
-----------------
Synchronous code executes in a blocking/sequential style.

@app.get("/sync")
def sync_route():
    return {"message":"Synchronous route"}

Asynchronous routes use async def.

@app.get("/async")
async def async_route():
    return {"message":"Asynchronous route"}

Async programming is especially useful for I/O-bound operations such as network requests and asynchronous database operations.

22. ASYNC ROUTES AND COROUTINES
-------------------------------
A function declared with async def is a coroutine function. FastAPI can execute async route functions.

@app.get("/async-demo")
async def async_demo():
    return {"message":"This is an async route"}

The main benefit appears when the route awaits asynchronous I/O, for example: result = await some_async_operation().

23. BACKGROUND TASKS
--------------------
BackgroundTasks allows a small task to run after the response is sent.

Example:
def write_notification(message: str):
    with open("notifications.txt", "a") as file:
        file.write(message + "\n")

@app.post("/notify")
def notify(message: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, message)
    return {"message":"Notification task added"}

Useful simple tasks include writing logs, saving non-critical information, or sending a simple notification. BackgroundTasks is not a replacement for a full distributed task queue.

24. DEPENDENCY INJECTION
------------------------
Dependency Injection means providing a route with a value or functionality that it depends on.

Example:
def get_current_user():
    return "Admin"

@app.get("/profile")
def profile(user=Depends(get_current_user)):
    return {"message":"Dependency Injection Example", "current_user":user}

FastAPI calls the dependency and passes its result to the route.

25. Depends()
-------------
Depends() tells FastAPI that a function parameter should be provided by another dependency function.

Dependencies can be used for authentication, authorization, common parameters, reusable validation, and shared logic.

26. DEPENDENCY WITH PARAMETERS
------------------------------
Dependencies can contain checks such as user roles.

Example concept:
def get_user_role():
    return "Admin"

@app.get("/admin")
def admin_route(role=Depends(get_user_role)):
    if role != "Admin":
        raise HTTPException(status_code=403, detail="Access denied")
    return {"message":"Welcome Admin"}

27. ENVIRONMENT CONFIGURATION
-----------------------------
Environment variables let configuration change without editing application code.

Example:
import os
APP_NAME = os.getenv("APP_NAME", "FastAPI Education Assistant")

Typical environment values include application name, debug settings, secret keys, database URLs, and token expiration settings. Secrets should not be hard-coded or committed to a public repository.

28. STRUCTURING LARGE FASTAPI PROJECTS
--------------------------------------
A small project can use main.py. As the project grows, code can be separated into modules.

Example:
project/
    app/
        main.py
        models.py
        routes.py
        dependencies.py
    data/
        syllabus.py
    uploads/
    requirements.txt
    README.md

The purpose is organization and maintainability. A student project should avoid unnecessary enterprise architecture.

============================================================
UNIT 4 — OAUTH2 AUTHENTICATION
============================================================

29. AUTHENTICATION VS AUTHORIZATION
-----------------------------------
Authentication answers: Who are you?
Authorization answers: What are you allowed to access?

Authentication verifies identity. Authorization decides permissions after identity is known.

30. OAUTH2
----------
OAuth2 is an authorization framework commonly used for securing APIs. FastAPI provides tools for OAuth2 flows.

OAuth2PasswordBearer is commonly used to read bearer access tokens.

from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

The tokenUrl identifies the endpoint where a client obtains a token.

31. OAUTH2 PASSWORD FLOW
------------------------
Simplified flow:
1. User sends username and password to login.
2. Server verifies credentials.
3. Server generates an access token.
4. Client stores the token.
5. Client sends the token on future requests.
6. Protected routes verify the token.

Header format:
Authorization: Bearer <access_token>

32. OAUTH2PASSWORDREQUESTFORM
-----------------------------
OAuth2PasswordRequestForm receives OAuth2 password-flow form data.

from fastapi.security import OAuth2PasswordRequestForm
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password
    return {"username": username}

OAuth2 password login uses form data rather than a normal JSON body.

33. JWT
-------
JWT means JSON Web Token. It is commonly used as an access token.

A JWT has three parts: header, payload, signature. They appear as header.payload.signature.

Payload claims can include subject, username, and expiration. A signature allows the server to verify token integrity.

A signed JWT is not automatically encrypted. Do not put sensitive secrets into a JWT payload assuming it is private.

34. CREATING A JWT
------------------
A library such as python-jose can create and verify JWTs.

Concept:
from jose import jwt
SECRET_KEY = "change-this-secret"
token = jwt.encode({"sub":"admin"}, SECRET_KEY, algorithm="HS256")

The secret key should be protected and normally supplied through environment configuration in real applications.

35. PROTECTED ROUTES
--------------------
An OAuth2 dependency can read the bearer token and validate it.

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    # verify token and return user
    ...

@app.get("/profile")
def profile(user=Depends(get_current_user)):
    return {"user": user}

36. AUTHENTICATION FLOW FOR THE RAG API
---------------------------------------
POST /login -> verify credentials -> generate JWT -> client receives token -> client sends Bearer token -> protected /ask verifies token -> RAG retrieval runs -> answer and source are returned.

============================================================
RAG API — EDUCATIONAL KNOWLEDGE RETRIEVAL
============================================================

37. WHAT IS RAG?
----------------
RAG means Retrieval-Augmented Generation. It combines retrieval of relevant information from a knowledge source with answer generation.

For a simple educational API, retrieval can first search syllabus notes and return the relevant information. This does not require a vector database or an external AI service.

38. KNOWLEDGE BASE
------------------
The knowledge base is the collection of information used by the retrieval system. For this project it can contain Unit 1–4 notes, definitions, examples, practical explanations, API concepts, and OAuth2 notes.

A simple Python list can store records such as:
{"unit":3,"topic":"Dependency Injection","content":"FastAPI uses Depends() to provide dependencies."}

A text file can also be used as the knowledge source.

39. SIMPLE RETRIEVAL
--------------------
A beginner-friendly retrieval method can:
1. Convert the question to lowercase.
2. Split it into words.
3. Compare words with topic content.
4. Count matching keywords.
5. Rank topics by score.
6. Return the highest-scoring relevant topic(s).

This is simple, explainable, and appropriate for demonstrating retrieval in a college project.

40. SEARCH ENDPOINT
-------------------
GET /search?query=dependency

The API can return matching topics, units, and content. The query is a query parameter, so this endpoint also demonstrates query parameters.

41. ASK ENDPOINT
----------------
POST /ask
Request:
{"question":"What is Depends() in FastAPI?"}

Possible response:
{"question":"What is Depends() in FastAPI?","answer":"Depends() is used by FastAPI to provide dependencies to route functions.","source":"Dependency Injection","unit":3}

The /ask route is the central educational RAG endpoint.

42. RAG SOURCE INFORMATION
--------------------------
A RAG response should identify the source topic so users can verify the answer. Useful fields are unit, topic, section, and source file.

============================================================
PRACTICAL FASTAPI CRUD
============================================================

43. CRUD
--------
CRUD means Create, Read, Update, Delete.

POST /topics — Create
GET /topics — Read all
GET /topics/{topic_id} — Read one
PUT /topics/{topic_id} — Update
DELETE /topics/{topic_id} — Delete

CRUD demonstrates REST methods and is useful in practical examinations.

44. IN-MEMORY DATA
------------------
A simple college project can store data temporarily in a Python list or dictionary. This avoids database complexity.

Example:
topics = []
next_id = 1

Limitation: data is lost when the server restarts.

45. CRUD ERROR HANDLING
-----------------------
When retrieving a topic by ID, check whether it exists. If not, return 404 Not Found. Similar checks should be used for invalid or duplicate operations.

============================================================
API RESPONSE DESIGN
============================================================

46. GOOD API RESPONSE
---------------------
Responses should be clear, structured, predictable, and useful.

Example:
{"message":"Topic created successfully","topic":{"id":1,"title":"FastAPI","unit":1}}

Error example:
{"detail":"Topic not found"}

47. RESPONSE STATUS CODES
-------------------------
POST creation: 201 Created.
GET success: 200 OK.
PUT success: 200 OK.
DELETE success: 200 OK or 204 No Content.
Missing resource: 404 Not Found.
Invalid input: commonly 422 Unprocessable Entity.
Missing/invalid authentication: 401 Unauthorized.
Insufficient permission: 403 Forbidden.

============================================================
PROJECT-SPECIFIC RAG API ROUTES
============================================================

48. SYSTEM ROUTES
-----------------
GET / — welcome message.
GET /health — API health/status.

49. SYLLABUS ROUTES
-------------------
GET /syllabus — syllabus information.
GET /units — available units.
GET /units/{unit_id} — topics in one unit.
GET /topics — all topics.
GET /topics/{topic_id} — one topic.

50. TOPIC CRUD ROUTES
---------------------
POST /topics — create topic.
PUT /topics/{topic_id} — update topic.
DELETE /topics/{topic_id} — delete topic.

51. SEARCH ROUTE
----------------
GET /search?query=fastapi — search the syllabus knowledge base and return matching topics.

52. ASK ROUTE
-------------
POST /ask — retrieve the most relevant syllabus information for a user question and return an educational answer plus source.

53. PROFILE ROUTE
-----------------
GET /profile — demonstrates Depends() and can be protected by OAuth2/JWT.

54. LOGIN ROUTE
---------------
POST /login — verifies demo credentials and returns an access token. A classroom project may use an in-memory user dictionary. Real applications should use secure password hashing and proper secret management.

55. ASYNC DEMO
--------------
GET /async-demo — demonstrates async def.

56. FILE ROUTES
---------------
POST /files/upload — uploads study material.
GET /files/download/{filename} — downloads stored material.

57. RESPONSE CLASS ROUTES
-------------------------
GET /response/json — demonstrates JSONResponse.
GET /response/html — demonstrates HTMLResponse.
File download demonstrates FileResponse.

============================================================
IMPORTANT DIFFERENCES
============================================================

58. PATH VS QUERY VS BODY
-------------------------
Path: GET /students/10 — identifies a specific resource.
Query: GET /students?course=AI — filters/searches.
Body: POST /students with JSON — sends structured creation/update data.

59. AUTHENTICATION VS DEPENDENCY INJECTION
------------------------------------------
Depends() is FastAPI's dependency injection mechanism. Authentication is a security process. Authentication can be implemented through a dependency, so Depends() is the mechanism while authentication is one use of it.

60. ASYNC VS BACKGROUND TASKS
-----------------------------
An async route is asynchronous and can await asynchronous operations. A BackgroundTask runs small follow-up work after the response is sent. They solve different problems.

61. JSON VS HTML RESPONSE
-------------------------
JSONResponse returns structured JSON. HTMLResponse returns HTML. Both are response classes and can be used by a backend without building a frontend application.

62. OAUTH2 VS JWT
----------------
OAuth2 is an authorization framework/flow. JWT is a token format. An API can use OAuth2 concepts with JWT access tokens. They are related but not identical.

============================================================
COMMON FASTAPI ERRORS
============================================================

63. 404 NOT FOUND
-----------------
The requested route or resource does not exist. Check URL, HTTP method, path parameters, and resource ID.

64. 422 VALIDATION ERROR
------------------------
Request data does not match the Pydantic model or declared parameter type. Check required fields, names, types, Enum values, and validators.

65. 401 UNAUTHORIZED
---------------------
Authentication is missing or invalid. Check Authorization header, bearer token, login process, and token validation.

66. 403 FORBIDDEN
-----------------
The user is authenticated but does not have permission for the operation.

67. IMPORT ERRORS
-----------------
Common causes include a missing package, wrong import, wrong virtual environment, or incorrect module name.

68. FILE UPLOAD ERROR
---------------------
If UploadFile/File does not work, install python-multipart because multipart form/file parsing requires it.

============================================================
SWAGGER TESTING GUIDE
============================================================

69. TESTING THROUGH /DOCS
-------------------------
Start the server with uvicorn main:app --reload and open /docs.

Testing steps:
1. Select an endpoint.
2. Click Try it out.
3. Enter path/query parameters or JSON body.
4. Execute.
5. Inspect status code and response.
6. Correct the request if validation fails.

For protected endpoints:
1. Call /login.
2. Obtain the access token.
3. Click Authorize in Swagger.
4. Enter the bearer token as required by the OAuth2 security scheme.
5. Call protected endpoints.

============================================================
BEST PRACTICES FOR THIS EDUCATIONAL PROJECT
============================================================

70. KEEP THE PROJECT SIMPLE
---------------------------
The goal is to demonstrate syllabus concepts clearly. A beginner-friendly project can use FastAPI, Pydantic, simple Python data structures, Depends(), BackgroundTasks, UploadFile, response classes, OAuth2/JWT, and simple retrieval. A vector database, LangChain, external LLM API, or complex repository architecture is not required for a basic syllabus demonstration.

71. USE CLEAR NAMING
--------------------
Prefer predictable resource names such as /topics, /topics/{topic_id}, and /search. Use meaningful function, model, and variable names.

72. VALIDATE INPUT
------------------
Use Pydantic models for request bodies, Enum for fixed choices, and custom validators for business rules. Do not assume incoming data is valid.

73. RETURN CORRECT STATUS CODES
------------------------------
Use 201 for successful creation, 404 for missing resources, 401 for missing/invalid authentication, 403 for insufficient permission, and 422 for validation errors.

74. HANDLE ERRORS
-----------------
Use HTTPException for predictable API errors and provide clear details such as "Syllabus topic not found".

75. DOCUMENT THE API
--------------------
FastAPI automatically creates OpenAPI documentation at /docs and /redoc. Good route names, request models, response models, and status codes make the generated documentation easier to understand.

============================================================
QUICK REVISION / CHEAT SHEET
============================================================

FastAPI app:
app = FastAPI()

GET:
@app.get("/items")
def get_items():
    return {"items": []}

POST:
@app.post("/items")
def create_item(item: Item):
    return item

Path parameter:
@app.get("/items/{item_id}")
def get_item(item_id: int):
    ...

Query parameter:
@app.get("/search")
def search(query: str):
    ...

Dependency:
@app.get("/profile")
def profile(user=Depends(get_current_user)):
    ...

HTTP error:
raise HTTPException(status_code=404, detail="Not found")

Async:
@app.get("/async-demo")
async def async_demo():
    ...

Background task:
background_tasks.add_task(function, argument)

File upload:
file: UploadFile = File(...)

File response:
return FileResponse("file.txt")

HTML response:
return HTMLResponse("<h1>Hello</h1>")

============================================================
KEY TERMS
============================================================

API — Application Programming Interface.
REST — architectural style for resource-oriented network APIs.
HTTP — protocol used for client-server communication.
FastAPI — Python framework for building APIs.
Pydantic — library for data validation and parsing.
BaseModel — Pydantic base class used to define structured models.
HTTPException — FastAPI class used to return HTTP errors.
Depends — FastAPI dependency injection function.
BackgroundTasks — FastAPI utility for small tasks after a response.
UploadFile — FastAPI type for uploaded files.
FileResponse — response class for returning files.
JSONResponse — response class for explicit JSON responses.
HTMLResponse — response class for HTML responses.
OAuth2 — authorization framework commonly used for API security.
JWT — JSON Web Token, commonly used as a bearer access token.
Bearer token — token sent using the Authorization header with Bearer scheme.
RAG — Retrieval-Augmented Generation.
Retrieval — finding relevant information from a knowledge base.
Knowledge base — collection of information used by a retrieval system.

============================================================
VIVA / INTERVIEW QUESTIONS
============================================================

Q: What is FastAPI?
A: FastAPI is a Python web framework for building APIs with automatic validation and OpenAPI documentation, and it supports synchronous and asynchronous routes.

Q: What is REST?
A: REST is an architectural style for designing APIs around resources and HTTP methods.

Q: Difference between GET and POST?
A: GET is generally used to retrieve data, while POST is generally used to create or submit data.

Q: What is a path parameter?
A: A value embedded in the URL path, such as /students/10.

Q: What is a query parameter?
A: A parameter supplied after ? in the URL, such as /students?course=AI.

Q: What is Pydantic?
A: Pydantic provides data validation and parsing using Python type definitions.

Q: What is a Pydantic model?
A: A class derived from BaseModel that defines expected data structure and types.

Q: What is Depends()?
A: Depends() is FastAPI's dependency injection mechanism.

Q: What is BackgroundTasks?
A: It allows small tasks to run after the API response is sent.

Q: What is async def?
A: It defines an asynchronous function that can await asynchronous operations.

Q: What is OAuth2?
A: OAuth2 is an authorization framework used to control access to resources.

Q: What is JWT?
A: JWT is a signed token format commonly used for carrying authentication claims.

Q: What is a bearer token?
A: A token sent in the Authorization header using the Bearer scheme.

Q: What is 404?
A: Not Found.

Q: What is 401?
A: Unauthorized; authentication is missing or invalid.

Q: What is 403?
A: Forbidden; the user is authenticated but lacks permission.

Q: What is 422 in FastAPI?
A: It commonly indicates that request data failed validation.

Q: What is Swagger?
A: Swagger UI is an interactive API documentation and testing interface generated from FastAPI's OpenAPI schema.

Q: What is ReDoc?
A: ReDoc is another interface for viewing OpenAPI documentation.

Q: What is RAG?
A: Retrieval-Augmented Generation retrieves relevant knowledge before producing or returning an answer.

Q: Why return a source in a RAG response?
A: It shows which knowledge-base topic was used and makes the answer easier to verify.

Q: Why use an in-memory list in a college project?
A: It keeps the project simple and demonstrates API concepts without a database. The limitation is that data is lost when the server restarts.

============================================================
PROJECT SUMMARY
============================================================

FastAPI Education Assistant — RAG API is an educational backend designed to demonstrate the API Development syllabus. It should cover REST API design, FastAPI routes, path/query/body parameters, Pydantic validation, nested models, Enum and custom validation, response models, HTTP status codes, HTTPException, JSON/HTML/file responses, file upload/download, async routes, BackgroundTasks, Depends(), environment configuration, project structure, OAuth2, JWT authentication, simple syllabus retrieval/RAG, and Swagger/ReDoc testing.

No separate frontend is required. Swagger (/docs) and ReDoc (/redoc) are automatically generated backend documentation/testing interfaces.

END OF NOTES

============================================================
NLP & RAG TESTING KNOWLEDGE BASE
============================================================

NLP NOTES — DETAILED KNOWLEDGE BASE FOR RAG / USER-INPUT TESTING
Purpose
This file is designed as a knowledge base for an NLP/RAG API. A user can type questions such as:
What is NLP?
Explain tokenization.
What is TF-IDF?
Difference between stemming and lemmatization.
Explain Naive Bayes for text classification.
What is VADER?
Explain LDA topic modeling.
What is perplexity?
Difference between RNN and LSTM.
What is attention?
What are BERT, GPT and RoBERTa? The retrieval model should return the most relevant section when the answer is covered by these notes.
COURSE OVERVIEW
The syllabus contains five units and approximately 60 total teaching hours.
UNIT I — FUNDAMENTALS OF NLP & TEXT PREPROCESSING
NATURAL LANGUAGE PROCESSING OVERVIEW
Natural Language Processing (NLP) is a field of Artificial Intelligence and Machine Learning that enables computers to process, understand, analyze and generate human language.
Human language is difficult for computers because it is ambiguous, context-dependent, variable, and often contains spelling mistakes, slang, abbreviations and incomplete sentences.
Main goals of NLP:
Convert human language into a form that computers can process.
Extract useful information from text or speech.
Understand meaning, intent and relationships.
Classify or predict properties of text.
Generate natural-language responses.
Common NLP applications:
Search engines
Chatbots and virtual assistants
Machine translation
Sentiment analysis
Spam detection
Text classification
Question answering
Named Entity Recognition
Document summarization
Information extraction
Speech-related systems
Recommendation and personalization
Autocomplete and text generation
Major challenges in NLP:
Ambiguity: one word or sentence can have multiple meanings.
Context: meaning depends on surrounding words.
Synonymy: different words can express similar meanings.
Polysemy: one word can have multiple related meanings.
Sarcasm and irony: literal words may not represent intended meaning.
Negation: "good" and "not good" have different meanings.
Spelling and grammar errors.
Slang, abbreviations and informal language.
Multilingual and code-mixed text.
Domain-specific terminology.
Long-range dependencies in sentences.
Lack of labeled training data for some languages/domains.
History of NLP: Early NLP systems relied heavily on manually written rules. Statistical NLP later used probabilities learned from data. Machine learning and deep learning improved classification and sequence processing. Modern NLP uses transformer architectures and pretrained language models such as BERT, GPT and RoBERTa.
TEXT CLEANING AND PREPROCESSING
Text preprocessing transforms raw text into a cleaner and more useful representation before analysis or model training.
Typical preprocessing pipeline: Raw text -> cleaning -> sentence segmentation -> word/token segmentation -> lowercasing -> punctuation/noise handling -> stopword handling -> stemming or lemmatization -> feature representation -> machine learning / NLP task
2.1 Lowercasing Lowercasing converts characters to lowercase.
Example: "FastAPI IS Useful" -> "fastapi is useful"
Advantages:
Reduces vocabulary size.
Treats "Python" and "python" as the same token.
Disadvantages:
May lose information where capitalization matters.
Named entities can sometimes be easier to detect using capitalization.
2.2 Tokenization Tokenization splits text into smaller units called tokens.
Word tokenization: "Natural language is useful." -> ["Natural", "language", "is", "useful"]
Sentence tokenization: "FastAPI is fast. NLP is useful." -> ["FastAPI is fast.", "NLP is useful."]
Tokenization is important because most NLP algorithms operate on tokens or sequences of tokens.
2.3 Stopword Removal Stopwords are very common words that may carry relatively little information for some tasks.
Examples in English: the, is, a, an, of, to, in, and
Example: "The student is learning NLP" may become: ["student", "learning", "NLP"]
Stopword removal can reduce noise and vocabulary size, but it should not always be used. Words such as "not" can be important for sentiment and meaning.
2.4 Stemming Stemming reduces words to a crude root-like form, obtaining a root word by cutting or removing word endings, prefixes or suffixes using rules.
Examples: playing -> play played -> play studies -> studi (depending on stemmer)
Advantages:
Fast.
Simple.
Reduces vocabulary.
Disadvantages:
May produce non-dictionary words.
Can remove too much or too little.
2.5 Lemmatization Lemmatization converts a word to its valid dictionary base form (lemma), usually using linguistic information.
Examples: am, are, is -> be better -> good (with suitable linguistic processing) running -> run (depending on context/tagging)
Advantages:
More linguistically meaningful than stemming.
Produces valid base forms more often.
Disadvantages:
Usually more computationally expensive.
May require part-of-speech information.
Stemming vs Lemmatization: Stemming is usually faster and rule-based; lemmatization is linguistically informed and usually more accurate for obtaining a meaningful base form.
REGULAR EXPRESSIONS IN TEXT CLEANING
A regular expression (regex) is a pattern used to find, extract, replace or validate text.
Common regex operations:
Search for patterns.
Remove unwanted characters.
Extract email addresses.
Find numbers.
Replace repeated whitespace.
Validate simple formats.
Examples: Pattern for digits: \d+ Pattern for whitespace: \s+ A simple email pattern can be used to find strings resembling user@example.com.
Example cleaning: Input: "Hello!!!   NLP 101 :)" A regex can remove or replace punctuation and repeated spaces depending on the task.
Regex is useful for:
URLs
email extraction
phone-number-like patterns
hashtags
mentions
punctuation cleanup
HTML-like noise
repeated spaces
Important limitation: Regex is pattern matching, not true language understanding. Complex linguistic meaning cannot be solved reliably with regex alone.
PART-OF-SPEECH (POS) TAGGING
Part-of-Speech tagging assigns a grammatical category to each word.
Common POS categories:
Noun (NN)
Verb (VB)
Adjective (JJ)
Adverb (RB)
Pronoun
Preposition
Determiner
Conjunction
Example: "The student writes code." The -> determiner student -> noun writes -> verb code -> noun
Why POS tagging is useful:
Helps lemmatization.
Helps syntactic analysis.
Helps information extraction.
Helps identify important grammatical patterns.
Helps distinguish word meanings in context.
A word may have different POS tags depending on context. For example, "book" can be a noun ("a book") or a verb ("book a ticket").
NAMED ENTITY RECOGNITION (NER)
Named Entity Recognition identifies real-world entities in text and assigns entity labels.
Common entity types:
PERSON
ORGANIZATION
LOCATION / GPE
DATE
TIME
MONEY
PRODUCT
EVENT
Example: "Sachin studies at ABC University in Nashik." Possible entities: Sachin -> PERSON ABC University -> ORGANIZATION Nashik -> LOCATION
Applications:
Information extraction
Search
Question answering
Resume analysis
News analysis
Customer support
Knowledge graph construction
POS vs NER: POS identifies grammatical roles such as noun or verb. NER identifies specific real-world entity categories such as person, organization or location.
SENTENCE SEGMENTATION AND WORD SEGMENTATION
Sentence segmentation divides a document into sentences.
Example: "NLP is useful. FastAPI can expose a model." -> two sentences.
Word segmentation divides a sentence into words/tokens.
For English, spaces make word segmentation relatively easy. Some languages do not use spaces between every word, making segmentation more difficult.
Why segmentation matters:
NLP models often process text sentence by sentence or token by token.
It supports parsing, classification, summarization and information extraction.
UNIT I QUICK QUESTIONS
Q: What is NLP? A: NLP is AI technology for processing, understanding, analyzing and generating human language.
Q: Why is preprocessing required? A: It reduces noise and converts raw text into a form suitable for NLP algorithms.
Q: What is tokenization? A: Splitting text into tokens such as words or sentences.
Q: What is the main difference between stemming and lemmatization? A: Stemming uses simpler reduction rules and may produce non-words; lemmatization uses linguistic knowledge to obtain a meaningful base form.
Q: What does POS tagging do? A: It assigns grammatical categories to words.
Q: What does NER do? A: It identifies named entities such as persons, organizations and locations.
UNIT II — TEXT REPRESENTATION AND VECTORIZATION
WHY TEXT MUST BE REPRESENTED NUMERICALLY
Machine learning algorithms generally require numerical input. Text must therefore be converted into vectors.
A vector is a list of numbers representing a word, sentence or document.
A good text representation should capture useful information while controlling dimensionality and noise.
Main methods in this unit:
Bag of Words
TF-IDF
Word2Vec
GloVe
Doc2Vec
PCA
t-SNE
Word similarity and analogy tasks
BAG OF WORDS (BoW)
Bag of Words represents a document using word occurrence or frequency while ignoring grammar and word order.
Suppose the vocabulary is: ["cat", "dog", "runs"]
Document 1: "cat runs"
Document 2: "dog runs"
Binary BoW representation: Document 1 -> [1, 0, 1] Document 2 -> [0, 1, 1]
Count-based BoW stores how many times each word appears.
Advantages:
Simple.
Easy to implement.
Effective for many basic classification tasks.
Disadvantages:
Ignores word order.
Produces high-dimensional sparse vectors.
Does not directly capture semantic similarity.
"dog bites man" and "man bites dog" can have the same word counts.
TERM FREQUENCY–INVERSE DOCUMENT FREQUENCY (TF-IDF)
TF-IDF assigns a larger weight to terms that are important in a document but not common across all documents.
TF = Term Frequency. A simple form is:
TF(t,d) = number of times term t occurs in document d / total number of terms in document d
IDF = Inverse Document Frequency.
A common smoothed form is: IDF(t) = log((N + 1)/(DF(t) + 1)) + 1
where: N = number of documents DF(t) = number of documents containing term t
TF-IDF: TF-IDF(t,d) = TF(t,d) × IDF(t)
Interpretation:
A word appearing many times in one document can get high TF.
A word appearing in nearly every document gets low IDF.
A specific, informative word can receive a high TF-IDF score.
Example: If "the" occurs in almost every document, its IDF is low. If "tokenization" appears in only a few documents, its IDF is higher.
TF-IDF is widely used in:
Document retrieval
Search
Text classification
Similarity matching
Simple RAG/retrieval systems
WORD EMBEDDINGS
Word embeddings represent words as dense numerical vectors. Words with similar meanings or usage can have similar vectors.
Unlike basic one-hot or BoW representations, embeddings can encode relationships between words.
10.1 Word2Vec Word2Vec is a neural word embedding approach.
Two important architectures:
CBOW (Continuous Bag of Words)
Skip-Gram
CBOW: Uses surrounding/context words to predict a target word.
Concept: context words -> target word
Skip-Gram: Uses a target word to predict surrounding/context words.
Concept: target word -> context words
CBOW is generally efficient for frequent words and can work well with large datasets. Skip-Gram can be useful for learning representations of less frequent words.
Word2Vec learns vectors based on word-context relationships.
GloVe
GloVe stands for Global Vectors for Word Representation.
GloVe learns word vectors using global word co-occurrence statistics from a corpus.
Main idea: Words that occur in similar contexts or have meaningful co-occurrence patterns can receive related vectors.
Word2Vec mainly uses local context prediction. GloVe uses global co-occurrence information.
DOCUMENT EMBEDDINGS — DOC2VEC
Doc2Vec extends the idea of Word2Vec to represent larger pieces of text such as paragraphs and documents.
A document vector can be used for:
Document classification
Document similarity
Information retrieval
Clustering
The goal is to learn a fixed-size vector representing the document.
DIMENSIONALITY REDUCTION
Text vectors can have hundreds or thousands of dimensions. Dimensionality reduction maps them to fewer dimensions.
Reasons:
Visualization
Faster computation
Noise reduction
Easier analysis
13.1 PCA — Principal Component Analysis PCA is a linear dimensionality reduction technique.
Basic idea:
Center the data.
Find directions of maximum variance.
Project data onto selected principal components.
If a dataset has 100 features, PCA might reduce it to 2 or 3 components for visualization.
Advantages:
Fast and widely used.
Preserves directions with high variance.
Useful for preprocessing and visualization.
Limitation: PCA is linear and may not capture complex nonlinear structures.
13.2 t-SNE t-SNE (t-distributed Stochastic Neighbor Embedding) is mainly used to visualize high-dimensional data in two or three dimensions.
It tries to preserve local neighborhood relationships.
Typical use: Plot word or document embeddings to see clusters.
Important caution: t-SNE plots are mainly exploratory visualizations. Distances between far-apart clusters should not automatically be interpreted as exact global relationships.
PCA vs t-SNE: PCA is a linear technique and is generally simpler/faster. t-SNE is a nonlinear visualization technique that focuses strongly on local neighborhoods.
WORD SIMILARITY AND ANALOGY TASKS
Word embeddings can be evaluated or explored using similarity.
Cosine similarity: cos(A,B) = (A · B) / (||A|| ||B||)
A value closer to 1 means vectors point in similar directions. A value around 0 indicates weak directional similarity for typical normalized vectors.
Analogy tasks test relationships such as: king - man + woman ≈ queen
The exact quality depends on the embedding model and training data.
UNIT II QUICK QUESTIONS
Q: What is BoW? A: A representation that records word occurrence/frequency while ignoring word order.
Q: What does TF-IDF measure? A: The importance of a term in a document relative to its occurrence across the document collection.
Q: What is CBOW? A: Word2Vec architecture that predicts a target word from context words.
Q: What is Skip-Gram? A: Word2Vec architecture that predicts context words from a target word.
Q: What is GloVe? A: A word embedding method based on global word co-occurrence statistics.
Q: What is Doc2Vec? A: A method for learning vector representations of documents.
Q: What is PCA? A: A linear dimensionality reduction technique based on principal directions of variance.
Q: What is t-SNE? A: A nonlinear technique mainly used to visualize high-dimensional embeddings.
UNIT III — TEXT CLASSIFICATION AND SENTIMENT ANALYSIS
TEXT CLASSIFICATION
Text classification assigns one or more predefined labels to text.
Examples:
spam vs not spam
positive vs negative
sports vs politics vs technology
complaint vs query vs feedback
Typical workflow:
Collect labeled text.
Clean/preprocess text.
Convert text to numerical vectors.
Split into training and test data.
Train classifier.
Predict labels.
Evaluate performance.
NAIVE BAYES FOR TEXT
Naive Bayes is a probabilistic classifier based on Bayes' theorem.
Bayes' theorem: P(C|X) = P(X|C)P(C) / P(X)
For text classification, the "naive" assumption is that features are conditionally independent given the class.
Common variant: Multinomial Naive Bayes is suitable for word-count or TF-IDF-like text features.
Advantages:
Simple.
Fast.
Works well for many text classification problems.
Effective with high-dimensional sparse features.
Limitations:
Independence assumption is often unrealistic.
May not capture complex relationships between words.
LOGISTIC REGRESSION FOR TEXT
Logistic Regression is a supervised classification algorithm that estimates class probabilities.
For binary classification: p = 1 / (1 + e^(-z))
where z is a weighted combination of input features.
With TF-IDF vectors, Logistic Regression can perform well for:
sentiment classification
spam detection
topic classification
Advantages:
Simple and interpretable.
Strong baseline.
Efficient for sparse text vectors.
SVM FOR TEXT CATEGORIZATION
Support Vector Machine (SVM) finds a decision boundary that separates classes while maximizing the margin.
Linear SVM is commonly used for high-dimensional sparse text features.
Advantages:
Strong performance in many text classification tasks.
Effective with high-dimensional TF-IDF data.
Works well when classes are reasonably separable.
Important terms:
Hyperplane: decision boundary.
Margin: distance between boundary and closest training examples.
Support vectors: examples that influence the boundary.
SENTIMENT ANALYSIS
Sentiment analysis determines the emotional or opinion-related polarity of text.
Common labels:
Positive
Negative
Neutral
Applications:
Product reviews
Social media analysis
Customer feedback
Survey analysis
Brand monitoring
Challenges:
Sarcasm
Negation
Context
Mixed sentiment
Domain-specific words
VADER
VADER stands for Valence Aware Dictionary and sEntiment Reasoner.
It is a lexicon/rule-based sentiment analysis method designed especially for social-media-like text.
VADER considers sentiment-related words and rules involving:
punctuation
capitalization
degree modifiers
contrast
negation-like patterns
Typical VADER output includes:
positive
negative
neutral
compound score
Compound score is a normalized overall sentiment score, commonly interpreted using conventional thresholds in example applications.
Advantages:
Fast.
No model training required for basic use.
Good for short informal text.
Limitation: It may be less effective when domain-specific context is very different from its lexicon/rules.
TEXTBLOB
TextBlob is a Python library providing simple NLP functionality, including sentiment analysis.
TextBlob sentiment can provide:
polarity
subjectivity
Polarity is generally on a negative-to-positive scale. Subjectivity represents how opinion-based or factual the text is.
Example: "I love this course." may have positive polarity.
VADER vs TextBlob: VADER is rule/lexicon oriented and designed particularly for social-media-style sentiment. TextBlob provides a simple sentiment interface based on its underlying NLP resources.
MODEL EVALUATION
A classifier should be evaluated on data not used for fitting.
Important metrics:
Confusion matrix: For binary classification:
True Positive (TP)
True Negative (TN)
False Positive (FP)
False Negative (FN)
Accuracy: Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision: Precision = TP / (TP + FP)
Precision answers: "Of the items predicted positive, how many were actually positive?"
Recall: Recall = TP / (TP + FN)
Recall answers: "Of the actual positive items, how many did the model find?"
F1 score: F1 = 2 × (Precision × Recall) / (Precision + Recall)
F1 balances precision and recall.
When classes are imbalanced, accuracy alone can be misleading. Precision, recall and F1 can provide more useful information.
CROSS-VALIDATION
Cross-validation estimates model performance by repeatedly splitting data into training and validation portions.
K-fold cross-validation:
Divide data into K folds.
Train on K-1 folds.
Validate on the remaining fold.
Repeat until every fold has been used for validation.
Average the scores.
Advantages:
Better use of limited data.
More stable performance estimate.
Important: Test data should ideally remain untouched until final evaluation.
HYPERPARAMETER TUNING
Hyperparameters are settings selected before or around model training rather than learned directly from the training examples.
Examples:
SVM C
Logistic Regression regularization strength
Naive Bayes smoothing parameter
n-gram range
vocabulary limits
Common tuning methods:
Grid search
Random search
Cross-validation-based search
Goal: Find settings that improve validation performance without overfitting.
PIPELINES USING SCIKIT-LEARN AND SPACY
A pipeline combines multiple processing steps.
Example scikit-learn text pipeline: raw text -> TfidfVectorizer -> LogisticRegression -> prediction
Benefits:
Keeps preprocessing and model together.
Reduces mistakes.
Makes training and prediction consistent.
Easier to evaluate and deploy.
spaCy provides NLP pipelines that can include:
tokenizer
POS tagger
dependency parser
NER
other components
UNIT III QUICK QUESTIONS
Q: Which classifier is often strong for TF-IDF text? A: Linear SVM and Logistic Regression are common strong baselines; Naive Bayes is also widely used.
Q: What is precision? A: The fraction of predicted positives that are actually positive.
Q: What is recall? A: The fraction of actual positives correctly identified.
Q: What is F1? A: The harmonic mean of precision and recall.
Q: What is VADER? A: A lexicon/rule-based sentiment analyzer designed especially for short informal text.
Q: What does TextBlob sentiment provide? A: Commonly polarity and subjectivity.
UNIT IV — TOPIC MODELING AND LANGUAGE MODELING
TOPIC MODELING
Topic modeling discovers latent themes or topics in a collection of documents without requiring predefined labels.
For example, a set of articles might contain hidden topics such as:
sports
politics
technology
education
A topic model can represent a document as a mixture of topics.
LDA — LATENT DIRICHLET ALLOCATION
LDA is a probabilistic topic-modeling algorithm.
Core intuition:
A document contains multiple topics.
Each topic contains a distribution of words.
A document's words are generated according to its topic mixture.
Example: A document about machine learning may contain: 70% machine-learning topic 20% programming topic 10% education topic
Important terms:
Topic distribution: how strongly a document is associated with topics.
Word distribution: which words are important to a topic.
Number of topics: a model setting chosen by the user.
Applications:
Document exploration
News clustering
Content discovery
Research-paper analysis
Limitations:
Topics may be difficult to interpret.
Number of topics often must be chosen.
Results depend on preprocessing and data.
NMF — NON-NEGATIVE MATRIX FACTORIZATION
NMF factorizes a non-negative document-term matrix into lower-dimensional non-negative matrices.
Conceptually: X ≈ W × H
Where: X = original document-term matrix W = document-topic representation H = topic-term representation
Because values are non-negative, the resulting components can often be interpreted as additive topics.
LDA vs NMF: LDA is probabilistic and generative. NMF is a matrix factorization method that can be applied to non-negative text feature matrices such as TF-IDF.
LANGUAGE MODELING
A language model estimates probabilities of sequences of words/tokens.
Example: Given "The student is learning", a language model estimates likely next tokens such as "NLP".
A language model can be used for:
Next-word prediction
Text generation
Autocomplete
Speech and translation components
Evaluation of sequence likelihood
N-GRAMS
An n-gram is a sequence of n consecutive tokens.
Unigram: "machine"
Bigram: "machine learning"
Trigram: "machine learning model"
Examples: Sentence: "NLP is useful" Unigrams: NLP, is, useful Bigrams: NLP is; is useful Trigram: NLP is useful
N-gram models estimate the next word using a limited history.
A bigram model approximates: P(w_i | w_1,...,w_{i-1}) ≈ P(w_i | w_{i-1})
A trigram model approximates: P(w_i | previous words) ≈ P(w_i | w_{i-2}, w_{i-1})
Advantages:
Simple.
Easy to understand.
Useful for introductory language modeling.
Limitations:
Limited context.
Data sparsity.
Vocabulary growth.
Cannot easily model long-range dependencies.
SMOOTHING TECHNIQUES
N-gram models can assign zero probability to unseen n-grams. Smoothing redistributes some probability mass to unseen events.
Examples:
Add-one (Laplace) smoothing
Add-k smoothing
Good-Turing smoothing
Kneser-Ney smoothing
Simple add-one idea: Add 1 to each count before calculating probabilities.
Smoothing prevents unseen sequences from automatically receiving probability zero.
PERPLEXITY
Perplexity is a common evaluation metric for language models.
Conceptually, perplexity measures how well a model predicts a sequence. Lower perplexity generally indicates better predictive performance on the same evaluation setup.
For a sequence of N tokens: PP = exp( - (1/N) × sum(log P(w_i)) )
Equivalent intuition: Perplexity is related to the inverse probability assigned to the observed sequence.
Important: Perplexity values should be compared only when the evaluation conditions, tokenization and datasets are comparable.
TEXT GENERATION USING MARKOV CHAINS
A Markov language model assumes that the next state depends on a limited recent history.
A first-order word Markov model may use: P(next_word | current_word)
Generation process:
Start with a word.
Find possible next words from training data.
Select one according to learned probabilities.
Repeat.
Advantages:
Simple.
Easy to implement.
Useful for demonstrating probabilistic text generation.
Limitations:
Short memory.
Repetitive or incoherent output.
Cannot capture long context effectively.
RNNs FOR LANGUAGE MODELING
Recurrent Neural Networks process sequences while maintaining a hidden state.
At each time step: current input + previous hidden state -> new hidden state -> output
RNNs can model sequence order better than basic bag-of-words methods.
Problem: Standard RNNs can suffer from vanishing or exploding gradients, making long-term dependencies difficult to learn.
UNIT IV QUICK QUESTIONS
Q: What is topic modeling? A: Unsupervised discovery of latent themes in a collection of documents.
Q: What is LDA? A: A probabilistic topic-modeling technique where documents are mixtures of topics and topics are distributions over words.
Q: What is NMF? A: A non-negative matrix factorization approach that can discover additive topic components from text matrices.
Q: What is an n-gram? A: A sequence of n consecutive tokens.
Q: Why is smoothing needed? A: To reduce the zero-probability problem for unseen n-grams.
Q: What does lower perplexity generally mean? A: Better predictive performance under the same evaluation conditions.
Q: What is a Markov-chain text generator? A: A generator that chooses the next state/token based on a limited previous history.
UNIT V — DEEP LEARNING & TRANSFORMERS IN NLP
RNN — RECURRENT NEURAL NETWORK
An RNN is a neural network designed for sequential data.
At time t: h_t = f(W_x x_t + W_h h_(t-1) + b)
where: x_t = current input h_t = current hidden state h_(t-1) = previous hidden state
RNNs reuse parameters across sequence positions.
Applications:
Sequence classification
Language modeling
Time-series-like sequence processing
Earlier-generation machine translation and speech systems
Main problem: Vanishing gradients can make learning long-range dependencies difficult.
LSTM — LONG SHORT-TERM MEMORY
LSTM is a specialized RNN designed to better preserve information over longer sequences.
LSTM uses gates:
Forget gate
Input gate
Output gate
Conceptually: Forget gate decides what old information to remove. Input gate decides what new information to store. Output gate decides what information to expose as hidden state.
LSTM is better than a basic RNN at handling many long-term dependency problems.
GRU — GATED RECURRENT UNIT
GRU is another gated recurrent architecture.
Main gates:
Update gate
Reset gate
GRU has a simpler structure than LSTM and often has fewer parameters.
LSTM vs GRU: LSTM has a more explicit cell state and three major gate mechanisms. GRU combines some mechanisms into a simpler architecture. Both can handle longer dependencies better than a basic RNN.
ATTENTION MECHANISM
Attention allows a model to focus on the most relevant parts of an input when producing an output.
Instead of treating every input position equally, the model calculates weights representing relevance.
Basic attention concept: Query + Keys -> attention scores Attention scores -> normalized weights Weights + Values -> context representation
Advantages:
Helps capture relevant context.
Makes long-range relationships easier to model.
Became a foundation of modern transformer architectures.
SEQ2SEQ MODELS
Sequence-to-sequence models map one sequence to another.
Examples:
Machine translation
Text summarization
Question answering
Dialogue systems
A traditional seq2seq architecture often has: Encoder -> Decoder
Encoder processes the input sequence. Decoder generates the output sequence.
Attention can help the decoder access different encoder states instead of relying only on one fixed representation.
TRANSFORMERS
Transformers are neural architectures based heavily on attention rather than recurrence.
Important components:
Self-attention
Multi-head attention
Positional information
Feed-forward networks
Residual connections
Layer normalization
Self-attention allows each token to consider other tokens in the sequence.
Basic idea: For each token, calculate Query, Key and Value representations.
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
where d_k is the key dimension.
Transformers can process sequence positions more parallelly than recurrent models during training and are highly effective for NLP.
BERT
BERT stands for Bidirectional Encoder Representations from Transformers.
BERT is an encoder-based pretrained transformer model.
Important idea: BERT learns contextual representations by considering information from both directions around a token.
BERT is useful for:
Text classification
Named Entity Recognition
Question answering
Sentence similarity
Token classification
BERT can be fine-tuned on a labeled task.
GPT
GPT stands for Generative Pre-trained Transformer.
GPT-style models are decoder-based autoregressive language models.
Main idea: Predict the next token based on previous context.
GPT models are commonly used for:
Text generation
Dialogue
Summarization
Question answering
Code generation
Other generative tasks
BERT vs GPT: BERT is primarily encoder-based and is designed for strong contextual understanding and representation. GPT is decoder-based and designed around autoregressive next-token generation.
RoBERTa
RoBERTa is a robustly optimized approach based on the BERT architecture.
It improves training methodology and can achieve strong language understanding performance.
RoBERTa is commonly used for:
Text classification
NER
Sentence-level NLP tasks
Other encoder-based applications
BERT vs RoBERTa: Both are encoder-based transformer models, but RoBERTa uses an improved training setup and different training choices.
TEXT CLASSIFICATION WITH HUGGING FACE TRANSFORMERS
Hugging Face Transformers provides pretrained transformer models and utilities.
Typical workflow:
Choose a pretrained tokenizer.
Tokenize text.
Convert tokens into model inputs.
Run the pretrained or fine-tuned model.
Obtain logits/predictions.
Convert predictions into labels.
A tokenizer can produce:
input IDs
attention masks
sometimes token type IDs
For classification, a model produces scores/logits for classes.
FINE-TUNING PRETRAINED NLP MODELS
Fine-tuning means starting from a pretrained model and training it further on a task-specific dataset.
Example: A pretrained transformer can be fine-tuned on:
positive/negative reviews
spam/ham
news categories
intent classification
General steps:
Prepare labeled dataset.
Split into train/validation/test sets.
Select pretrained tokenizer and model.
Tokenize the dataset.
Train with an appropriate optimizer and learning rate.
Validate.
Evaluate on held-out test data.
Benefits:
Reuses knowledge learned from large corpora.
Often needs less task-specific data than training from scratch.
Strong performance on many NLP tasks.
ETHICAL CONSIDERATIONS IN LANGUAGE MODELS
NLP systems can create or amplify problems if data and deployment are not handled carefully.
Important concerns:
Bias and fairness
Privacy
Misinformation
Hallucinated information
Copyright and data provenance
Toxic or harmful output
Security
Transparency
Evaluation across different populations and languages
Good practices:
Use appropriate and lawful data.
Evaluate bias and errors.
Protect sensitive information.
Clearly communicate system limitations.
Monitor deployed systems.
Avoid treating generated text as automatically factual.
UNIT V QUICK QUESTIONS
Q: Why can basic RNNs struggle with long sequences? A: Vanishing/exploding gradients can make long-term dependencies difficult to learn.
Q: What is the purpose of LSTM gates? A: They control what information is forgotten, stored and exposed.
Q: What is a GRU? A: A simpler gated recurrent neural architecture using update/reset mechanisms.
Q: What is attention? A: A mechanism that assigns different importance to input elements based on relevance.
Q: What is a transformer? A: A neural architecture built primarily around attention mechanisms and capable of highly parallel sequence processing.
Q: What is BERT? A: An encoder-based bidirectional pretrained transformer used for language understanding tasks.
Q: What is GPT? A: A decoder-based autoregressive transformer family designed around next-token prediction and generation.
Q: What is RoBERTa? A: An optimized BERT-style encoder model with improved training methodology.
IMPORTANT COMPARISONS
STEMMING VS LEMMATIZATION
Stemming:
Simple reduction.
Often rule-based.
Faster.
May create non-words.
Does not necessarily use grammatical context.
Lemmatization:
Finds a linguistic base form.
Usually uses vocabulary and linguistic information.
More meaningful.
Usually slower.
BoW VS TF-IDF
BoW:
Uses counts or binary occurrence.
Simple.
Common words may dominate.
Does not explicitly reduce weight of globally common words.
TF-IDF:
Uses term frequency and inverse document frequency.
Downweights terms common across documents.
Often better for information retrieval and classification.
WORD2VEC VS GLOVE
Word2Vec:
Predictive approach.
CBOW and Skip-Gram.
Learns from context prediction.
GloVe:
Uses global word co-occurrence statistics.
Learns vectors from a co-occurrence-based objective.
PCA VS t-SNE
PCA:
Linear.
Fast.
Good for general dimensionality reduction.
Can be used before downstream models.
t-SNE:
Nonlinear.
Mainly visualization.
Preserves local neighborhoods.
Can be slower and sensitive to settings.
NAIVE BAYES VS LOGISTIC REGRESSION VS SVM
Naive Bayes:
Probabilistic.
Very fast.
Strong baseline for sparse text.
Uses conditional-independence assumption.
Logistic Regression:
Discriminative classifier.
Provides class probabilities in common binary/multiclass settings.
Strong with TF-IDF.
Easy baseline.
SVM:
Margin-based classifier.
Linear SVM is strong for high-dimensional sparse text.
Does not naturally provide calibrated probabilities without additional processing.
VADER VS TEXTBLOB
VADER:
Lexicon/rule-based.
Good for short informal/social text.
Includes a compound sentiment score.
TextBlob:
Simple Python NLP interface.
Sentiment commonly expressed through polarity and subjectivity.
LDA VS NMF
LDA:
Probabilistic topic model.
Documents are mixtures of topics.
Topics are distributions over words.
NMF:
Matrix factorization.
Works naturally with non-negative matrices such as TF-IDF.
Produces additive components.
RNN VS LSTM VS GRU
RNN:
Basic recurrent architecture.
Simple.
Can struggle with long dependencies.
LSTM:
Uses forget/input/output gates.
Better at long-term dependencies.
More complex.
GRU:
Uses update/reset gates.
Simpler than LSTM.
Often efficient while retaining gated memory behavior.
RNN/SEQ2SEQ VS TRANSFORMER
RNN/seq2seq:
Processes sequence recurrently.
Natural sequential computation.
Earlier standard for many sequence tasks.
Transformer:
Uses self-attention.
More parallelizable during training.
Strong at capturing relationships across a sequence.
Foundation of many modern pretrained NLP models.
CORE FORMULAS
TF-IDF
TF-IDF(t,d) = TF(t,d) × IDF(t)
One common smoothed IDF: IDF(t) = log((N+1)/(DF(t)+1)) + 1
COSINE SIMILARITY
cos(A,B) = (A·B) / (||A|| ||B||)
Used to compare vector direction and commonly used for document/text similarity.
PRECISION
Precision = TP / (TP + FP)
RECALL
Recall = TP / (TP + FN)
F1 SCORE
F1 = 2 × Precision × Recall / (Precision + Recall)
ACCURACY
Accuracy = (TP + TN) / (TP + TN + FP + FN)
BAYES THEOREM
P(C|X) = P(X|C)P(C) / P(X)
LANGUAGE MODEL PERPLEXITY
PP = exp(-(1/N) × sum(log P(w_i)))
SELF-ATTENTION
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
NLP MODEL TESTING / RAG TESTING GUIDE
PURPOSE OF TESTING
A retrieval or NLP model should be tested with questions that are:
Directly answered by the notes.
Paraphrases of note content.
Comparison questions.
Definition questions.
Formula questions.
Questions whose answer is NOT in the notes.
The model should retrieve relevant content for covered questions and avoid pretending to know unsupported information.
EXAMPLE USER INPUTS THAT SHOULD FIND AN ANSWER
What is Natural Language Processing?
List common applications of NLP.
What are the challenges of NLP?
Explain tokenization with an example.
What is stopword removal?
What is stemming?
What is lemmatization?
Difference between stemming and lemmatization.
What are regular expressions used for in NLP?
What is POS tagging?
What is NER?
Difference between POS tagging and NER.
What is Bag of Words?
What is TF-IDF?
Explain the TF-IDF formula.
Why does TF-IDF reduce the importance of common words?
What is Word2Vec?
Explain CBOW.
Explain Skip-Gram.
Difference between CBOW and Skip-Gram.
What is GloVe?
What is Doc2Vec?
What is PCA?
What is t-SNE?
Difference between PCA and t-SNE.
What is cosine similarity?
What is text classification?
Explain Naive Bayes for text.
What is Logistic Regression used for in NLP?
What is SVM text categorization?
What is sentiment analysis?
What is VADER?
What is TextBlob sentiment analysis?
What is a confusion matrix?
Define precision, recall and F1.
Why is accuracy sometimes misleading?
What is cross-validation?
What is hyperparameter tuning?
What is a scikit-learn pipeline?
What is topic modeling?
Explain LDA.
Explain NMF.
Difference between LDA and NMF.
What is a language model?
What is an n-gram?
What is bigram and trigram?
Why is smoothing required?
What is perplexity?
What is a Markov-chain text generator?
What is an RNN?
Why do RNNs have vanishing-gradient problems?
What is LSTM?
What are LSTM gates?
What is GRU?
Difference between LSTM and GRU.
What is attention?
What is seq2seq?
What is a transformer?
Explain self-attention.
What is BERT?
What is GPT?
What is RoBERTa?
Difference between BERT and GPT.
What is Hugging Face Transformers?
What is fine-tuning?
What are ethical concerns in language models?
PARAPHRASED TEST INPUTS
These are useful for testing whether a retrieval model understands similar wording.
"How do computers deal with human language?" -> NLP overview. "Why do we split a sentence into tokens?" -> Tokenization. "How is a root word obtained by cutting endings?" -> Stemming. "How do I get the dictionary form of a word?" -> Lemmatization. "How can I identify people and organizations in text?" -> NER. "How do I convert documents into numerical word vectors using frequency?" -> BoW / TF-IDF. "How does TF-IDF decide which words matter?" -> TF-IDF. "What is the Word2Vec method that predicts the missing center word?" -> CBOW. "Which Word2Vec approach predicts nearby words from one word?" -> Skip-Gram. "How can I visualize high-dimensional text vectors in two dimensions?" -> PCA/t-SNE. "Which metric uses TP, FP and FN to evaluate a classifier?" -> Precision/Recall/F1. "How can I discover hidden themes in documents?" -> Topic modeling. "How can a language model be measured using uncertainty-like score?" -> Perplexity. "Which recurrent model has forget and input gates?" -> LSTM. "Which model uses self-attention instead of recurrence?" -> Transformer. "Which transformer is mainly encoder based?" -> BERT. "Which transformer is autoregressive?" -> GPT.
OUT-OF-SCOPE / UNKNOWN QUESTION TESTS
These questions are intentionally not directly answered by this syllabus:
What is the current stock price of a company?
What is today's weather?
Who is the current president of a country?
What is the latest version of a software library?
What is the exact tuition fee of a university?
What is the user's personal password?
What is a topic not discussed anywhere in these notes?
Expected RAG behavior: If the notes do not contain enough information, the system should say that the topic was not found in the provided notes instead of inventing an answer.
TEST CASE CATEGORIES FOR A RAG API
A. Exact-match test: Question closely matches a sentence in the notes.
B. Paraphrase test: Question uses different wording but asks the same concept.
C. Comparison test: Question asks for differences between two concepts.
D. Formula test: Question asks for a formula such as TF-IDF, F1 or cosine similarity.
E. Multi-concept test: Question asks about two related concepts, such as "BERT vs GPT."
F. Unknown test: Question asks something outside the notes.
G. Temporary-training test: Upload a temporary text file containing a new concept, then ask a question about that concept.
TEMPORARY TRAINING CONCEPT
Temporary training means replacing/retraining the in-memory retrieval knowledge for the current application process using a user-provided text file.
Example:
Upload a text file containing: "FastAPI is a Python web framework for building APIs."
The NLP retrieval model trains on the uploaded text.
Ask: "What is FastAPI?"
The model retrieves the relevant uploaded content.
Important behavior:
Temporary training should not overwrite the permanent notes unless explicitly designed to do so.
Restarting the application can restore the default knowledge base.
This is useful for testing how retrieval behaves on a new small dataset.
HOW A SIMPLE TF-IDF RAG RETRIEVER WORKS
A basic retrieval model can work as follows:
Training:
Read notes.
Split notes into documents/sections.
Tokenize text.
Calculate document frequency.
Calculate IDF values.
Create TF-IDF vectors for documents.
Store vectors in memory.
Question answering:
Receive user question.
Tokenize the question.
Create its TF-IDF vector using the trained vocabulary/IDF.
Calculate cosine similarity between the question and note sections.
Rank sections by similarity.
Return the best matching relevant section.
If similarity is too low, return an "information not found" response.
This is retrieval-based NLP rather than a large generative language model. It is simple, explainable and suitable for demonstrating TF-IDF, vectorization and similarity.
EXAMPLE RETRIEVAL FLOW
Question: "What is the difference between stemming and lemmatization?"
Relevant note concepts:
Stemming reduces words using simpler rules.
Lemmatization obtains a linguistically meaningful base form.
Stemming may produce non-words.
Lemmatization is generally more linguistically informed.
Expected answer: The system should retrieve the Stemming vs Lemmatization section or closely related Unit I content.
EXAMPLE UNKNOWN FLOW
Question: "Explain quantum computing hardware."
If quantum computing is not in the knowledge base, the system should not manufacture an NLP answer. A safe retrieval response is: "I could not find this topic in the provided notes."
NLP STUDY CHEAT SHEET
Unit I: NLP -> language processing Tokenization -> split text into tokens Lowercasing -> normalize case Stopword removal -> remove selected common words Stemming -> crude root reduction Lemmatization -> linguistic base form Regex -> pattern-based text processing POS -> grammatical role NER -> named entity category Segmentation -> split sentences/words
Unit II: BoW -> word occurrence/count vector TF-IDF -> importance of term in document collection Word2Vec -> predictive word embeddings CBOW -> context predicts target Skip-Gram -> target predicts context GloVe -> global co-occurrence embeddings Doc2Vec -> document embeddings PCA -> linear dimensionality reduction t-SNE -> nonlinear visualization Cosine similarity -> vector similarity
Unit III: Naive Bayes -> probabilistic classifier Logistic Regression -> discriminative classifier SVM -> maximum-margin classifier VADER -> lexicon/rule sentiment TextBlob -> simple NLP/sentiment Precision -> predicted-positive correctness Recall -> actual-positive coverage F1 -> balance of precision and recall Confusion matrix -> TP/TN/FP/FN Cross-validation -> repeated train/validation splits Hyperparameter tuning -> choose model settings Pipeline -> chained preprocessing/model steps
Unit IV: Topic modeling -> discover hidden themes LDA -> probabilistic topic model NMF -> non-negative matrix factorization Language model -> sequence probability N-gram -> n consecutive tokens Smoothing -> handle unseen sequences Perplexity -> language-model evaluation Markov chain -> limited-history generation RNN -> recurrent sequence model
Unit V: RNN -> recurrent neural network LSTM -> gated RNN with cell state GRU -> simpler gated RNN Attention -> weighted focus on relevant inputs Seq2Seq -> sequence-to-sequence mapping Transformer -> attention-based architecture BERT -> encoder-based bidirectional pretrained model GPT -> decoder-based autoregressive model RoBERTa -> optimized BERT-style model Hugging Face -> transformer model ecosystem/tools Fine-tuning -> adapt pretrained model to a task Ethics -> bias, privacy, misinformation, safety, transparency
SAMPLE VIVA QUESTIONS AND ANSWERS
Q1. Why is NLP difficult? A1. Human language is ambiguous, contextual, variable and can contain sarcasm, negation, spelling errors, slang and domain-specific terminology.
Q2. Why convert text into vectors? A2. Most machine learning algorithms operate on numerical data, so text must be represented numerically.
Q3. Why is TF-IDF useful? A3. It gives higher importance to terms that are frequent in a document but relatively uncommon across the document collection.
Q4. What is the difference between BoW and embeddings? A4. BoW is usually sparse and mainly represents word occurrence, while embeddings are dense vectors designed to capture relationships and semantic/contextual information.
Q5. Why use cosine similarity? A5. It measures similarity between vector directions and is widely useful for comparing text vectors.
Q6. Why is F1 useful? A6. F1 combines precision and recall and is useful when both false positives and false negatives matter.
Q7. What is overfitting? A7. Overfitting occurs when a model learns training data too specifically and performs poorly on unseen data.
Q8. What is cross-validation? A8. It repeatedly divides data into training and validation folds to estimate model performance more reliably.
Q9. What is the purpose of smoothing? A9. It prevents unseen n-grams from receiving zero probability.
Q10. Why did transformers become important? A10. Self-attention allows transformers to model relationships among tokens effectively and supports highly parallel training compared with recurrent processing.
Q11. What is fine-tuning? A11. Fine-tuning adapts a pretrained model to a particular downstream task using task-specific data.
Q12. What is the difference between a retrieval system and a generative model? A12. A retrieval system selects relevant information from an existing knowledge base, while a generative model creates new text based on learned patterns and context. A simple TF-IDF RAG system can retrieve relevant note content without using a large generative model.
FINAL TESTING CHECKLIST
For a syllabus-based NLP/RAG API, test at least:
One question from Unit I.
One paraphrased Unit I question.
One TF-IDF question from Unit II.
One embedding question from Unit II.
One classification question from Unit III.
One metric/formula question from Unit III.
One LDA or NMF question from Unit IV.
One perplexity/n-gram question from Unit IV.
One RNN/LSTM question from Unit V.
One transformer question from Unit V.
One BERT/GPT comparison.
One unknown question.
Temporary training with a custom text file.
Ask a question about the temporary content.
Check model status after training.
Restart the API and verify default notes can be loaded again.
KEY IDEA FOR THE PROJECT
The NLP syllabus and the FastAPI backend can be connected through a simple retrieval workflow:
User question -> FastAPI /ask endpoint -> text preprocessing -> TF-IDF vectorization -> cosine similarity -> retrieve relevant NLP notes -> return answer/source/score
Temporary testing: User uploads .txt -> /train/temporary -> model retrains in memory -> user asks a question -> /ask retrieves from temporary knowledge
This architecture demonstrates practical NLP concepts while remaining simple enough to understand, test and explain in a viva."""

def get_notes_text() -> str:
    """Returns the knowledge base notes text."""
    return NOTES_TEXT

def train_notes_model(model):
    """
    Trains a NotesMLModel instance directly using NOTES_TEXT from notes.py.
    """
    return model.train(NOTES_TEXT)

if __name__ == "__main__":
    print(f"notes.py loaded successfully. Total notes length: {len(NOTES_TEXT):,} characters.")
