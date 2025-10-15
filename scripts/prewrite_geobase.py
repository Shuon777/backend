import time
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.coordinates_finder import fetch_and_draw
from infrastructure.geo_db_store import get_place


BAIKAL_PLACES_FILE = os.path.join(os.path.dirname(__file__), "baikal_places.json")

with open(BAIKAL_PLACES_FILE, "r", encoding="utf-8") as f:
    places = json.load(f)

success = 0
for place in places:
    if get_place(place):
        print(f"✅ Уже есть: {place}")
        continue

    features = fetch_and_draw(place)
    if features:
        print(f"✅ Добавлено: {place}")
        success += 1
    else:
        print(f"⚠️ Не найдено: {place}")

    time.sleep(1.5)  

print(f"\n🏁 Готово: {success} из {len(places)} мест добавлены в БД.")
