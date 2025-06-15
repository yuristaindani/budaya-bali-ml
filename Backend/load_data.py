import os
import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

import requests
from bs4 import BeautifulSoup
import urllib.parse

# File paths
DATA_FILE_INDONESIA = Path("data/budaya_bali_lengkap_coba.json")
DATA_FILE_ENGLISH = Path("data/artikel_budaya_bali_inggris_coba.json")
FAISS_INDEX_INDONESIA = "faiss_index_indonesia_coba"
FAISS_INDEX_ENGLISH = "faiss_index_english_coba"

# ---------- SCRAPING UTILS ----------

SCRAPING_CONFIGS = [
    # Indonesia
    {
        "name": "Pura",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 8),
        "page_url": lambda base_url, page: f"{base_url}/id/Pura" if page == 1 else f"{base_url}/id/Pura?page={page}"
    },
    {
        "name": "Desa Adat",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base_url, page: f"{base_url}/id/Desa-Adat-Bali" if page == 1 else f"{base_url}/id/Desa-Adat-Bali?page={page}"
    },
    {
        "name": "Tradisi Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base_url, page: f"{base_url}/id/Tradisi-Bali" if page == 1 else f"{base_url}/id/Tradisi-Bali?page={page}"
    },
    {
        "name": "Kearifan Lokal",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base_url, page: f"{base_url}/id/kearifan-lokal-Bali" if page == 1 else f"{base_url}/id/kearifan-lokal-Bali?page={page}"
    },
    {
        "name": "Alam Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 8),
        "page_url": lambda base_url, page: f"{base_url}/id/Alam-Bali" if page == 1 else f"{base_url}/id/Alam-Bali?page={page}"
    },
    {
        "name": "Seni Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base_url, page: f"{base_url}/id/seni-bali" if page == 1 else f"{base_url}/id/seni-bali?page={page}"
    },
    {
        "name": "Cerita Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base_url, page: f"{base_url}/id/Cerita-Bali" if page == 1 else f"{base_url}/id/Cerita-Bali?page={page}"
    },
    {
        "name": "Usadha Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 7),
        "page_url": lambda base_url, page: f"{base_url}/id/Usadha-Bali" if page == 1 else f"{base_url}/id/Usadha-Bali?page={page}"
    },
    # English
    {
        "name": "Temples",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 9),
        "page_url": lambda base_url, page: f"{base_url}/Temples" if page == 1 else f"{base_url}/Temples?page={page}"
    },
    {
        "name": "Traditional Village",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Traditional-Village" if page == 1 else f"{base_url}/Bali-Traditional-Village?page={page}"
    },
    {
        "name": "Tradition",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Tradition" if page == 1 else f"{base_url}/Bali-Tradition?page={page}"
    },
    {
        "name": "Local Wisdom",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Local-Wisdom" if page == 1 else f"{base_url}/Bali-Local-Wisdom?page={page}"
    },
    {
        "name": "Nature",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base_url, page: f"{base_url}/Balinese-Nature" if page == 1 else f"{base_url}/Balinese-Nature?page={page}"
    },
    {
        "name": "Arts",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Arts" if page == 1 else f"{base_url}/Bali-Arts?page={page}"
    },
    {
        "name": "Stories",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Stories" if page == 1 else f"{base_url}/Bali-Stories?page={page}"
    },
    {
        "name": "Medical",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 7),
        "page_url": lambda base_url, page: f"{base_url}/Bali-Traditional-Medical" if page == 1 else f"{base_url}/Bali-Traditional-Medical?page={page}"
    },
]

def scrape_budayabali(category_config):
    """Scrape a single category and return list of artikel dicts."""
    artikel_data = []
    base_url = category_config["base_url"]
    for page in category_config["pages"]:
        page_url = category_config["page_url"](base_url, page)
        print(f"🔍 Memproses halaman: {page_url}")
        try:
            response = requests.get(page_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('div', class_='post-item')
            if not articles:
                print(f"🚫 Tidak ada artikel di halaman {page}.")
                continue
            for i, article in enumerate(articles):
                try:
                    link_tag = article.find('a', href=True)
                    article_url = urllib.parse.urljoin(base_url, link_tag['href']) if link_tag else None
                    title = ""
                    isi_lengkap = ""
                    img_url = ""
                    if article_url:
                        artikel_res = requests.get(article_url, timeout=10)
                        artikel_soup = BeautifulSoup(artikel_res.content, 'html.parser')
                        title_tag = artikel_soup.find('h1', class_='post-title')
                        title = title_tag.get_text(strip=True) if title_tag else f"Artikel_{len(artikel_data)+1}"
                        content_div = artikel_soup.find('div', class_='post-content')
                        if content_div:
                            paragraphs = content_div.find_all(['p', 'h3', 'li'])
                            isi_lengkap = "\n".join(p.get_text(strip=True) for p in paragraphs)
                        img_tag = artikel_soup.select_one('div.post-image img')
                        if img_tag and 'src' in img_tag.attrs:
                            img_url = urllib.parse.urljoin(base_url, img_tag['src'])
                    artikel_data.append({
                        'Judul': title,
                        'Link Artikel': article_url,
                        'Isi Lengkap': isi_lengkap,
                        'Link Gambar': img_url
                    })
                    print(f"[✓] Artikel disimpan: {title}")
                    time.sleep(1)
                except Exception as e:
                    print(f"[✗] Gagal proses artikel: {e}")
        except Exception as e:
            print(f"[✗] Gagal mengambil halaman {page}: {e}")
    return artikel_data

def auto_scrape_and_save_json(lang):
    """Scrape all configs for a language and save to merged .json."""
    all_data = []
    for config in SCRAPING_CONFIGS:
        if config['lang'] == lang:
            print(f"=== Scraping: {config['name']} ({lang}) ===")
            data = scrape_budayabali(config)
            all_data.extend(data)
    # Save merged
    if lang == "id":
        out_file = DATA_FILE_INDONESIA
    else:
        out_file = DATA_FILE_ENGLISH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Semua data {lang} selesai! Total {len(all_data)} artikel disimpan ke '{out_file}'.")

# ---------- FAISS SETUP UTILS ----------

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
    """Setup FAISS indexes for both languages, scrapes if needed."""
    # Scrape Indonesian dataset if file missing or user requests
    if not DATA_FILE_INDONESIA.exists() or os.environ.get("FORCE_SCRAPE_ID") == "1":
        print(f"🔄 Auto-scraping data Indonesia...")
        auto_scrape_and_save_json("id")
    # Scrape English dataset if file missing or user requests
    if not DATA_FILE_ENGLISH.exists() or os.environ.get("FORCE_SCRAPE_EN") == "1":
        print(f"🔄 Auto-scraping data English...")
        auto_scrape_and_save_json("en")
    
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