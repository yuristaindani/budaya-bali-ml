from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import query_rag_multilang
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Budaya Bali RAG Dual Language API",
    description="RAG system that uses Indonesian dataset for Indonesian questions and English dataset for other languages",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    question: str

class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: list
    dataset_used: str = None
    input_language: str = None

@app.get("/")
async def root():
    return {
        "message": "Budaya Bali RAG Dual Language API",
        "version": "2.0.0",
        "description": "Uses Indonesian dataset for Indonesian questions, English dataset for other languages"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/ask", response_model=QueryResponse)
async def query_api(input: QueryInput):
    try:
        print(f"\n=== REQUEST DITERIMA ===")
        print(f"Pertanyaan: {input.question}")
        
        # Query the dual language RAG system
        result = query_rag_multilang(input.question)
        
        print(f"\n=== RESPONS YANG DIKIRIM ===")
        print(f"Dataset yang digunakan: {result.get('dataset_used', 'unknown')}")
        print(f"Bahasa input: {result.get('input_language', 'unknown')}")
        print(f"Jawaban: {result.get('answer', 'Tidak ada jawaban')}")
        print(f"Jumlah sumber: {len(result.get('sources', []))}")
        
        return {
            "success": True,
            "answer": result.get("answer", "Maaf, saya tidak dapat menemukan jawaban."),
            "sources": result.get("sources", []),
            "dataset_used": result.get("dataset_used", "unknown"),
            "input_language": result.get("input_language", "unknown")
        }
        
    except Exception as e:
        print(f"\n=== ERROR ===")
        traceback.print_exc()
        logger.error(f"Error processing question: {str(e)}")
        
        return {
            "success": False,
            "answer": f"Maaf, terjadi kesalahan: {str(e)}",
            "sources": [],
            "dataset_used": "error",
            "input_language": "unknown"
        }

@app.get("/info")
async def get_system_info():
    """Get information about the RAG system"""
    return {
        "system": "Dual Language RAG for Budaya Bali",
        "datasets": {
            "indonesian": "budaya_bali_lengkap_coba.json",
            "english": "artikel_budaya_bali_inggris_coba.json"
        },
        "logic": {
            "indonesian_questions": "Uses Indonesian dataset",
            "other_languages": "Uses English dataset with translation"
        },
        "supported_languages": "Auto-detected, with special handling for Indonesian"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Budaya Bali RAG Dual Language API...")
    print("📚 Indonesian questions → Indonesian dataset")
    print("🌐 Other languages → English dataset with translation")
    uvicorn.run(app, host="0.0.0.0", port=8000)