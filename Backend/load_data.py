import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.docstore.document import Document

from scrape_budayabali import run_full_scraping   

load_dotenv()

DATA_FILE_INDONESIA = Path("data/budaya_bali_lengkap_coba.json")
DATA_FILE_ENGLISH   = Path("data/artikel_budaya_bali_inggris_coba.json")
FAISS_INDEX_INDONESIA = "faiss_index_indonesia_cohere"
FAISS_INDEX_ENGLISH   = "faiss_index_english_cohere"

def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

def docs_from_rows(rows: list[dict]) -> list[Document]:
    docs = []
    for r in rows:
        if r.get("Isi Lengkap") and r.get("Judul") and r.get("Link Artikel"):
            docs.append(
                Document(
                    page_content=r["Isi Lengkap"],
                    metadata={
                        "title": r["Judul"],
                        "url": r["Link Artikel"],
                        "image": r.get("Link Gambar", ""),
                    },
                )
            )
    return docs

def build_faiss(docs: list[Document], out_dir: str):
    print(f"Meng-embedding {len(docs)} dokumen...")
    embed = CohereEmbeddings(
        model="embed-multilingual-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )

    batch = 50
    db = None
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
        print(f"Batch {i // batch + 1}/{-(-len(docs) // batch)}")
        db = FAISS.from_documents(chunk, embed) if db is None else db.add_documents(chunk) or db
        time.sleep(5)

    db.save_local(out_dir)
    print(f"Index FAISS tersimpan di '{out_dir}'")

def setup_dual_indexes():
    print("Menjalankan scraping sebelum indexing...\n")
    run_full_scraping()
    print("\nScraping selesai. Mulai proses indexing.\n")

    if DATA_FILE_INDONESIA.exists():
        rows_id = load_json(DATA_FILE_INDONESIA)
        docs_id = docs_from_rows(rows_id)
        if docs_id:
            build_faiss(docs_id, FAISS_INDEX_INDONESIA)
    else:
        print(f"File {DATA_FILE_INDONESIA} tidak ditemukan.")

    if DATA_FILE_ENGLISH.exists():
        rows_en = load_json(DATA_FILE_ENGLISH)
        docs_en = docs_from_rows(rows_en)
        if docs_en:
            build_faiss(docs_en, FAISS_INDEX_ENGLISH)
    else:
        print(f"File {DATA_FILE_ENGLISH} tidak ditemukan.")

if __name__ == "__main__":
    setup_dual_indexes()
    print("\nKedua index FAISS siap digunakan.")
