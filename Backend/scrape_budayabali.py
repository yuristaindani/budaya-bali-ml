import json
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_FILE_INDONESIA = Path("data/budaya_bali_lengkap_coba.json")
DATA_FILE_ENGLISH   = Path("data/artikel_budaya_bali_inggris_coba.json")

# Konfigurasi kategori
SCRAPING_CONFIGS = [
    # Indonesia 
    {
        "name": "Pura",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 8),
        "page_url": lambda base, p: f"{base}/id/Pura" if p == 1 else f"{base}/id/Pura?page={p}",
    },
    {
        "name": "Desa Adat",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base, p: f"{base}/id/Desa-Adat-Bali" if p == 1 else f"{base}/id/Desa-Adat-Bali?page={p}",
    },
    {
        "name": "Tradisi Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base, p: f"{base}/id/Tradisi-Bali" if p == 1 else f"{base}/id/Tradisi-Bali?page={p}",
    },
    {
        "name": "Kearifan Lokal",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base, p: f"{base}/id/kearifan-lokal-Bali" if p == 1 else f"{base}/id/kearifan-lokal-Bali?page={p}",
    },
    {
        "name": "Alam Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 8),
        "page_url": lambda base, p: f"{base}/id/Alam-Bali" if p == 1 else f"{base}/id/Alam-Bali?page={p}",
    },
    {
        "name": "Seni Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base, p: f"{base}/id/seni-bali" if p == 1 else f"{base}/id/seni-bali?page={p}",
    },
    {
        "name": "Cerita Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base, p: f"{base}/id/Cerita-Bali" if p == 1 else f"{base}/id/Cerita-Bali?page={p}",
    },
    {
        "name": "Usadha Bali",
        "lang": "id",
        "base_url": "https://budayabali.com",
        "pages": range(1, 7),
        "page_url": lambda base, p: f"{base}/id/Usadha-Bali" if p == 1 else f"{base}/id/Usadha-Bali?page={p}",
    },
    #  English
    {
        "name": "Temples",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 9),
        "page_url": lambda base, p: f"{base}/Temples" if p == 1 else f"{base}/Temples?page={p}",
    },
    {
        "name": "Traditional Village",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base, p: f"{base}/Bali-Traditional-Village" if p == 1 else f"{base}/Bali-Traditional-Village?page={p}",
    },
    {
        "name": "Tradition",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base, p: f"{base}/Bali-Tradition" if p == 1 else f"{base}/Bali-Tradition?page={p}",
    },
    {
        "name": "Local Wisdom",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 5),
        "page_url": lambda base, p: f"{base}/Bali-Local-Wisdom" if p == 1 else f"{base}/Bali-Local-Wisdom?page={p}",
    },
    {
        "name": "Nature",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base, p: f"{base}/Balinese-Nature" if p == 1 else f"{base}/Balinese-Nature?page={p}",
    },
    {
        "name": "Arts",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 4),
        "page_url": lambda base, p: f"{base}/Bali-Arts" if p == 1 else f"{base}/Bali-Arts?page={p}",
    },
    {
        "name": "Stories",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 6),
        "page_url": lambda base, p: f"{base}/Bali-Stories" if p == 1 else f"{base}/Bali-Stories?page={p}",
    },
    {
        "name": "Medical",
        "lang": "en",
        "base_url": "https://budayabali.com",
        "pages": range(1, 7),
        "page_url": lambda base, p: f"{base}/Bali-Traditional-Medical" if p == 1 else f"{base}/Bali-Traditional-Medical?page={p}",
    },
]

def scrape_budayabali(category_config):
    hasil = []
    base_url = category_config["base_url"]
    for page in category_config["pages"]:
        page_url = category_config["page_url"](base_url, page)
        print(f"Memproses halaman: {page_url}")
        try:
            res = requests.get(page_url, timeout=10)
            soup = BeautifulSoup(res.content, "html.parser")
            articles = soup.find_all("div", class_="post-item")
            if not articles:
                print(f"Tidak ada artikel di halaman {page}.")
                continue
            for art in articles:
                try:
                    a_tag = art.find("a", href=True)
                    art_url = urllib.parse.urljoin(base_url, a_tag["href"]) if a_tag else None
                    title = ""
                    full_text = ""
                    img_url = ""
                    if art_url:
                        art_res = requests.get(art_url, timeout=10)
                        art_soup = BeautifulSoup(art_res.content, "html.parser")
                        title_node = art_soup.find("h1", class_="post-title")
                        title = title_node.get_text(strip=True) if title_node else "Tanpa Judul"
                        content_div = art_soup.find("div", class_="post-text mt-4") or art_soup.find("div", class_="post-content")
                        if content_div:
                            paras = content_div.find_all(["p", "h3", "li"])
                            full_text = "\n".join(p.get_text(strip=True) for p in paras if p.get_text(strip=True))
                        img_tag = art_soup.select_one("div.post-image img")
                        if img_tag and img_tag.get("src"):
                            img_url = urllib.parse.urljoin(base_url, img_tag["src"])
                    hasil.append(
                        {
                            "Judul": title,
                            "Link Artikel": art_url,
                            "Isi Lengkap": full_text,
                            "Link Gambar": img_url,
                        }
                    )
                    print(f"[✓] Artikel disimpan: {title}")
                    time.sleep(1)
                except Exception as err:
                    print(f"[✗] Gagal proses artikel: {err}")
        except Exception as err:
            print(f"[✗] Gagal mengambil halaman {page}: {err}")
    return hasil

def auto_scrape_and_save_json(lang):
    kumpulan = []
    for cfg in SCRAPING_CONFIGS:
        if cfg["lang"] == lang:
            print(f"=== Scraping kategori {cfg['name']} ({lang}) ===")
            kumpulan.extend(scrape_budayabali(cfg))

    out_file = DATA_FILE_INDONESIA if lang == "id" else DATA_FILE_ENGLISH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(kumpulan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSemua data {lang} selesai. {len(kumpulan)} artikel tersimpan di '{out_file}'")

def run_full_scraping():
    auto_scrape_and_save_json("id")
    auto_scrape_and_save_json("en")

if __name__ == "__main__":
    run_full_scraping()
