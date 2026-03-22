import logging
import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from app.config import BASE_DIR, RESOURCES_DIST_PATH, IMAGES_DIR, MAPS_DIR, DOMAIN, REDIS_HOST, REDIS_PORT, REDIS_DB
from app.services import geo, slot_val, search_service, relational_service  # импортируем для инициализации
from app.utils import init_redis
from app.routes import register_blueprints

load_dotenv()

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)

    # Инициализация Redis
    init_redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

    # Регистрация блюпринтов
    register_blueprints(app)

    @app.route("/")
    def home():
        return "SalutBot API works!"

    return app