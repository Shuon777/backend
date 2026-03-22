import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent  # корень проекта
RESOURCES_DIST_PATH = str(BASE_DIR / "json_files" / "resources_dist.json")
IMAGES_DIR = "images"
MAPS_DIR = os.getenv("MAPS_DIR")
DOMAIN = os.getenv("PUBLIC_BASE_URL")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")

EMBEDDING_MODEL_PATH = str(BASE_DIR / "embedding_models" / "bge-m3")
FAISS_INDEX_PATH = str(BASE_DIR / "knowledge_base_scripts" / "Vector" / "faiss_index")