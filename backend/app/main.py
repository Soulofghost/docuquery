from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.database import init_db
from app.routers import corpus, query

app = FastAPI(
    title="DocuQuery RAG API",
    description="Production-Grade Retrieval-Augmented Generation Document QA Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize schema
init_db()

# Mount Routers
app.include_router(corpus.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DocuQuery RAG Pipeline",
        "version": "1.0.0",
        "vector_engine": "Subword TF-IDF Cosine Embedding",
        "guardrails": "Active"
    }

# Mount Frontend
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
