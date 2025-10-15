import json
import os

MAPS_REGISTRY_PATH = "/var/www/salut_bot/json_files/maps_store.json"

if not os.path.exists(MAPS_REGISTRY_PATH):
    with open(MAPS_REGISTRY_PATH, "w") as f:
        json.dump({}, f)

def load_registry():
    with open(MAPS_REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(data):
    with open(MAPS_REGISTRY_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_map_links(map_id):
    registry = load_registry()
    return registry.get(map_id, {})

def set_map_links(map_id, links):
    registry = load_registry()
    registry[map_id] = links
    save_registry(registry)