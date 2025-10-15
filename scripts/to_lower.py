import json

db_path = "/var/www/salut_bot/json_files/maps_store.json"

with open(db_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Приведение ключей к нижнему регистру
new_data = {k.lower(): v for k, v in data.items()}

with open(db_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Ключи приведены к нижнему регистру.")
