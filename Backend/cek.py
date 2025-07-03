import json

with open('data/budaya_bali_lengkap_coba.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"Jumlah data: {len(data)}")