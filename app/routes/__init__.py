from flask import Blueprint
import logging

from app.routes.error_log import error_log_bp
from app.routes.database import database_bp
from app.routes.attractions import attractions_bp
from app.routes.polygon import polygon_bp
from app.routes.area import area_bp
from app.routes.images import images_bp
from app.routes.description import description_bp
from app.routes.coordinates import coordinates_bp
from app.routes.species import species_bp
from app.routes.faiss import faiss_bp

def register_blueprints(app):
    app.register_blueprint(error_log_bp)
    app.register_blueprint(database_bp)
    app.register_blueprint(attractions_bp)
    app.register_blueprint(polygon_bp)
    app.register_blueprint(area_bp)
    app.register_blueprint(images_bp)
    app.register_blueprint(description_bp)
    app.register_blueprint(coordinates_bp)
    app.register_blueprint(species_bp)
    app.register_blueprint(faiss_bp)