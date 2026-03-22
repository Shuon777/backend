import logging
from flask import Blueprint, request, jsonify
from infrastructure.db_utils_for_search import Slot_validator

species_bp = Blueprint('species', __name__)
logger = logging.getLogger(__name__)

@species_bp.route("/find_species_with_description", methods=["POST"])
def find_species_with_description():
    data = request.get_json()
    name = data.get("name")
    limit = data.get("limit", 1500)
    offset = data.get("offset", 0)
    
    logger.info(f"POST /find_species_with_description - name: {name}, limit: {limit}, offset: {offset}")
    
    if not name:
        return jsonify({
            "status": "error",
            "message": "Параметр 'name' обязателен",
            "used_objects": [],
            "not_used_objects": []
        }), 400
    
    slot_val = Slot_validator()
    result = slot_val.find_species_with_description(name, limit, offset)
    
    used_objects = []
    not_used_objects = []
    
    if result.get("status") == "success" and result.get("results"):
        for species in result["results"]:
            used_objects.append({
                "name": species.get("name", name),
                "type": "biological_entity"
            })
    else:
        not_used_objects.append({
            "name": name,
            "type": "biological_entity" 
        })
    
    result["used_objects"] = used_objects
    result["not_used_objects"] = not_used_objects
    
    return jsonify(result)