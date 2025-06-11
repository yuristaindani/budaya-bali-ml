# import os
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from pathlib import Path
# import json

# DATA_FILE = Path("data/artikel_budaya_bali_inggris.json")
# FAISS_FILE = "faiss_index"

# def load_json_data(filepath: Path) -> list[dict]:
#     with filepath.open("r", encoding="utf-8") as file:
#         return json.load(file)

# def prepare_documents(data: list[dict]) -> list[Document]:
#     docs = []
#     for item in data:
#         if all(k in item for k in ["Isi Lengkap", "Judul", "Link Artikel"]):
#             docs.append(
#                 Document(
#                     page_content=item["Isi Lengkap"],
#                     metadata={
#                         "title": item["Judul"],
#                         "url": item["Link Artikel"],
#                         "image": item.get("Link Gambar", "")
#                     }
#                 )
#             )
#     return docs

# def load_and_persist_faiss(documents: list[Document], faiss_file: str):
#     try:
#         print(f"Memulai proses embedding dan penyimpanan ke FAISS ({faiss_file}) ...")
#         embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = FAISS.from_documents(
#             documents,
#             embedding=embedding,
#         )
#         vectordb.save_local(faiss_file)
#         print("✅ FAISS index created and saved successfully ...")
#         print("You can now use this FAISS index for retrieval.")
#     except Exception as e:
#         print(f"❌ ERROR: {e}")

# if __name__ == "__main__":
#     if Path(FAISS_FILE).exists():
#         print("FAISS index already exists. Reloading is not necessary.")
#     else:
#         if not DATA_FILE.exists():
#             print(f"❌ Data file not found: {DATA_FILE}")
#         else:
#             print("Loading data from JSON and saving to FAISS...")
#             data = load_json_data(DATA_FILE)
#             documents = prepare_documents(data)
#             print(f"Loaded {len(documents)} valid records.")
#             if len(documents) == 0:
#                 print("❌ Tidak ada dokumen valid yang ditemukan di data.")
#             else:
#                 load_and_persist_faiss(documents, FAISS_FILE)


# KODE DUA
# import os
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from pathlib import Path
# import json

# DATA_FILE = Path("data/budaya_bali_lengkap.json")  # Ganti ke dataset Indonesia
# FAISS_FILE = "faiss_index_indonesia"  # Nama baru untuk index bahasa Indonesia

# def load_json_data(filepath: Path) -> list[dict]:
#     with filepath.open("r", encoding="utf-8") as file:
#         return json.load(file)

# def prepare_documents(data: list[dict]) -> list[Document]:
#     docs = []
#     for item in data:
#         # Pastikan field sesuai dengan dataset Indonesia
#         if all(k in item for k in ["Isi Lengkap", "Judul", "Link Artikel"]):
#             docs.append(
#                 Document(
#                     page_content=item["Isi Lengkap"],  # Konten bahasa Indonesia
#                     metadata={
#                         "title": item["Judul"],
#                         "url": item["Link Artikel"],
#                         "image": item.get("Link Gambar", "")
#                     }
#                 )
#             )
#     return docs

# def load_and_persist_faiss(documents: list[Document], faiss_file: str):
#     try:
#         print(f"Memulai proses embedding dan penyimpanan ke FAISS ({faiss_file}) ...")
#         embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = FAISS.from_documents(
#             documents,
#             embedding=embedding,
#         )
#         vectordb.save_local(faiss_file)
#         print("✅ FAISS index created and saved successfully ...")
#         print("You can now use this FAISS index for retrieval.")
#     except Exception as e:
#         print(f"❌ ERROR: {e}")

# if __name__ == "__main__":
#     if Path(FAISS_FILE).exists():
#         print("FAISS index sudah ada. Tidak perlu membuat ulang.")
#     else:
#         if not DATA_FILE.exists():
#             print(f"❌ File data tidak ditemukan: {DATA_FILE}")
#         else:
#             print("Memuat data dari JSON dan menyimpan ke FAISS...")
#             data = load_json_data(DATA_FILE)
#             documents = prepare_documents(data)
#             print(f"Memuat {len(documents)} dokumen valid.")
#             if len(documents) == 0:
#                 print("❌ Tidak ada dokumen valid yang ditemukan.")
#             else:
#                 load_and_persist_faiss(documents, FAISS_FILE)


# KODE TIGA
# import os
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.docstore.document import Document
# from pathlib import Path
# import json

# DATA_FILE = Path("data/budaya_bali_lengkap.json")  # Dataset bahasa Indonesia
# FAISS_FILE = "faiss_index_indonesia"  # Nama file index FAISS

# def load_json_data(filepath: Path) -> list[dict]:
#     with filepath.open("r", encoding="utf-8") as file:
#         return json.load(file)

# def prepare_documents(data: list[dict]) -> list[Document]:
#     docs = []
#     for item in data:
#         if all(k in item for k in ["Isi Lengkap", "Judul", "Link Artikel"]):
#             docs.append(
#                 Document(
#                     page_content=item["Isi Lengkap"],  # Konten bahasa Indonesia
#                     metadata={
#                         "title": item["Judul"],
#                         "url": item["Link Artikel"],
#                         "image": item.get("Link Gambar", "")
#                     }
#                 )
#             )
#     return docs

# def load_and_persist_faiss(documents: list[Document], faiss_file: str):
#     try:
#         print(f"Memulai proses embedding dan penyimpanan ke FAISS ({faiss_file}) ...")
#         embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = FAISS.from_documents(
#             documents,
#             embedding=embedding,
#         )
#         vectordb.save_local(faiss_file)
#         print("✅ FAISS index created and saved successfully ...")
#     except Exception as e:
#         print(f"❌ ERROR: {e}")

# if __name__ == "__main__":
#     if Path(FAISS_FILE).exists():
#         print("FAISS index sudah ada. Tidak perlu membuat ulang.")
#     else:
#         if not DATA_FILE.exists():
#             print(f"❌ File data tidak ditemukan: {DATA_FILE}")
#         else:
#             print("Memuat data dari JSON dan menyimpan ke FAISS...")
#             data = load_json_data(DATA_FILE)
#             documents = prepare_documents(data)
#             print(f"Memuat {len(documents)} dokumen valid.")
#             load_and_persist_faiss(documents, FAISS_FILE)

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pathlib import Path
import json

# File paths
DATA_FILE_INDONESIA = Path("data/budaya_bali_lengkap.json")
DATA_FILE_ENGLISH = Path("data/artikel_budaya_bali_inggris.json")
FAISS_INDEX_INDONESIA = "faiss_index_indonesia"
FAISS_INDEX_ENGLISH = "faiss_index_english"

def load_json_data(filepath: Path) -> list[dict]:
    """Load JSON data from file"""
    with filepath.open("r", encoding="utf-8") as file:
        return json.load(file)

def prepare_documents(data: list[dict]) -> list[Document]:
    """Prepare documents for FAISS indexing"""
    docs = []
    for item in data:
        if all(k in item for k in ["Isi Lengkap", "Judul", "Link Artikel"]):
            docs.append(
                Document(
                    page_content=item["Isi Lengkap"],
                    metadata={
                        "title": item["Judul"],
                        "url": item["Link Artikel"],
                        "image": item.get("Link Gambar", "")
                    }
                )
            )
    return docs

def load_and_persist_faiss(documents: list[Document], faiss_file: str, language: str):
    """Create and save FAISS index"""
    try:
        print(f"Memulai proses embedding dan penyimpanan ke FAISS ({faiss_file}) untuk bahasa {language}...")
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(
            documents,
            embedding=embedding,
        )
        vectordb.save_local(faiss_file)
        print(f"✅ FAISS index untuk bahasa {language} berhasil dibuat dan disimpan...")
    except Exception as e:
        print(f"❌ ERROR saat membuat index {language}: {e}")

def setup_dual_faiss_indexes():
    """Setup FAISS indexes for both languages"""
    
    # Setup Indonesian index
    if Path(FAISS_INDEX_INDONESIA).exists():
        print("FAISS index Indonesia sudah ada.")
    else:
        if not DATA_FILE_INDONESIA.exists():
            print(f"❌ File data Indonesia tidak ditemukan: {DATA_FILE_INDONESIA}")
        else:
            print("Loading data Indonesia dari JSON dan menyimpan ke FAISS...")
            data_indonesia = load_json_data(DATA_FILE_INDONESIA)
            documents_indonesia = prepare_documents(data_indonesia)
            print(f"Loaded {len(documents_indonesia)} dokumen Indonesia yang valid.")
            if len(documents_indonesia) == 0:
                print("❌ Tidak ada dokumen Indonesia yang valid ditemukan.")
            else:
                load_and_persist_faiss(documents_indonesia, FAISS_INDEX_INDONESIA, "Indonesia")
    
    # Setup English index
    if Path(FAISS_INDEX_ENGLISH).exists():
        print("FAISS index English sudah ada.")
    else:
        if not DATA_FILE_ENGLISH.exists():
            print(f"❌ File data English tidak ditemukan: {DATA_FILE_ENGLISH}")
        else:
            print("Loading data English dari JSON dan menyimpan ke FAISS...")
            data_english = load_json_data(DATA_FILE_ENGLISH)
            documents_english = prepare_documents(data_english)
            print(f"Loaded {len(documents_english)} dokumen English yang valid.")
            if len(documents_english) == 0:
                print("❌ Tidak ada dokumen English yang valid ditemukan.")
            else:
                load_and_persist_faiss(documents_english, FAISS_INDEX_ENGLISH, "English")

if __name__ == "__main__":
    setup_dual_faiss_indexes()
    print("\n🎉 Setup selesai! Kedua index FAISS siap digunakan.")