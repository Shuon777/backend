import logging
import time
from flask import Blueprint, request, jsonify
from app.services import search_service

images_bp = Blueprint('images', __name__)
logger = logging.getLogger(__name__)

@images_bp.route("/search_images_by_features", methods=["POST"])
def search_images_by_features():
    """
    Поиск изображений по признакам из feature_data
    Можно искать как по виду, так и только по признакам
    """
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")
    
    debug_info = {
        "timestamp": time.time(),
        "steps": []
    }
    
    try:
        data = request.get_json()
        species_name = data.get("species_name")
        features = data.get("features", {})
        if "fruits_present" not in features:
            features["fruits_present"] = "нет"
            
        if not species_name and not features:
            response = {
                "error": "Необходимо указать species_name или features",
                "used_objects": [],
                "not_used_objects": []
            }
            return jsonify(response), 400
        
        logger.info(f"🔍 /search_images_by_features - получен запрос с параметрами:")
        logger.info(f"   - species_name: {data.get('species_name')}")
        logger.info(f"   - features: {data.get('features', {})}")
        logger.info(f"   - query_params: debug_mode={debug_mode}, in_stoplist={in_stoplist}")
        
        debug_info["parameters"] = {
            "species_name": species_name,
            "features": features,
            "in_stoplist": in_stoplist,
            "timestamp": time.time()
        }
        
        safe_images = []
        stoplisted_images = []
        result = None
        
        if species_name:
            result = search_service.search_images_by_features(
                species_name=species_name,
                features=features
            )
            
            if result.get("status") == "success" and "images" in result:
                safe_images = []
                stoplisted_images = []
                
                for image in result["images"]:
                    feature_data = image.get("features", {})
                    image_in_stoplist = feature_data.get("in_stoplist")
                    
                    try:
                        requested_level = int(in_stoplist)
                        if image_in_stoplist is None or int(image_in_stoplist) <= requested_level:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                    except (ValueError, TypeError):
                        if image_in_stoplist is None or int(image_in_stoplist) <= 1:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                
                result["images"] = safe_images
                result["count"] = len(safe_images)
                result["in_stoplist_filter_applied"] = True
                result["in_stoplist_level"] = in_stoplist
                result["stoplisted_count"] = len(stoplisted_images)
            
            used_objects = []
            not_used_objects = []
            
            if result.get("status") == "success" and result.get("images"):
                used_objects.append({
                    "name": species_name,
                    "type": "biological_entity",
                    "images_count": len(result["images"])
                })
            
            result["used_objects"] = used_objects
            result["not_used_objects"] = not_used_objects
            
            if debug_mode:
                debug_info["search_type"] = "with_species"
                debug_info["synonyms_used"] = result.get("synonyms_used", {})
                debug_info["database_query"] = {
                    "species_conditions": result.get("species_conditions", []),
                    "feature_conditions": list(features.keys())
                }
                debug_info["stoplist_filter"] = {
                    "total_before_filter": len(result.get("images", [])),
                    "safe_after_filter": len(safe_images),
                    "stoplisted_count": len(stoplisted_images)
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                result["used_objects"] = []
                result["not_used_objects"] = []
                return jsonify(result), 404
            return jsonify(result)
        
        else:
            result = search_service.relational_service.search_images_by_features_only(
                features=features
            )
            
            if result.get("status") == "success" and "images" in result:
                safe_images = []
                stoplisted_images = []
                
                for image in result["images"]:
                    feature_data = image.get("features", {})
                    image_in_stoplist = feature_data.get("in_stoplist")
                    
                    try:
                        requested_level = int(in_stoplist)
                        if image_in_stoplist is None or int(image_in_stoplist) <= requested_level:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                    except (ValueError, TypeError):
                        if image_in_stoplist is None or int(image_in_stoplist) <= 1:
                            safe_images.append(image)
                        else:
                            stoplisted_images.append(image)
                            logger.info(f"Исключено изображение с in_stoplist={image_in_stoplist}: {image.get('title', 'Без названия')}")
                
                result["images"] = safe_images
                result["count"] = len(safe_images)
                result["in_stoplist_filter_applied"] = True
                result["in_stoplist_level"] = in_stoplist
                result["stoplisted_count"] = len(stoplisted_images)
            
            used_objects = []
            not_used_objects = []
            
            if result.get("status") == "success" and result.get("images"):
                unique_species = {}
                for image in result["images"]:
                    species = image.get("species_name")
                    if species and species not in unique_species:
                        unique_species[species] = {
                            "name": species,
                            "type": "biological_entity",
                            "images_count": 0
                        }
                    if species:
                        unique_species[species]["images_count"] += 1
                
                used_objects = list(unique_species.values())
            
            result["used_objects"] = used_objects
            result["not_used_objects"] = not_used_objects
            
            if debug_mode:
                debug_info["search_type"] = "features_only"
                debug_info["database_query"] = {
                    "feature_conditions": list(features.keys())
                }
                debug_info["stoplist_filter"] = {
                    "total_before_filter": len(result.get("images", [])),
                    "safe_after_filter": len(safe_images),
                    "stoplisted_count": len(stoplisted_images)
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                result["used_objects"] = []
                result["not_used_objects"] = []
                return jsonify(result), 404
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Ошибка поиска изображений по признакам: {str(e)}")
        error_response = {
            "status": "error",
            "message": f"Ошибка при поиске изображений: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500