import logging
import os
from pathlib import Path
import sys
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from app.config import BASE_DIR, RESOURCES_DIST_PATH, IMAGES_DIR, MAPS_DIR, DOMAIN, REDIS_HOST, REDIS_PORT, REDIS_DB
from app.services import geo, slot_val, search_service, relational_service
from app.utils import init_redis
from app.routes import register_blueprints
load_dotenv()

logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
logging.getLogger('search_api').setLevel(logging.DEBUG)
logging.getLogger('search_api.adapters').setLevel(logging.DEBUG)
logging.getLogger('search_api.use_cases').setLevel(logging.DEBUG)
logging.getLogger('search_api.services').setLevel(logging.DEBUG)
logging.getLogger('search_api.infrastructure').setLevel(logging.DEBUG)
from search_api.routes.search import search_bp
from search_api.config import SearchConfig
from search_api.infrastructure import RedisCache

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.INFO)

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(search_bp)
    init_redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    
    search_config = SearchConfig.from_env()
    search_redis = RedisCache(host=search_config.redis_host, port=search_config.redis_port, db=search_config.redis_db)
    if not search_redis.ping():
        app.logger.warning("Search API Redis connection failed")
    
    app.config['SEARCH_CONFIG'] = search_config
    app.config['SEARCH_REDIS'] = search_redis

    register_blueprints(app)

    @app.route("/")
    def home():
        return "SalutBot API works!"

    return app