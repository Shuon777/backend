import logging
import time
from flask import Blueprint, request, jsonify
from app.services import search_service, geo
from app.utils import generate_cache_key, get_cached_result, set_cached_result

coordinates_bp = Blueprint('coordinates', __name__)
logger = logging.getLogger(__name__)

@coordinates_bp.route("/get_coords", methods=["POST"])
def api_get_coords():
    data = request.get_json()
    name = data.get("name")
    
    logger.info(f"🔍 /get_coords - получен запрос:")
    logger.info(f"   - name: {name}")
    
    if not name:
        return jsonify({
            "status": "error", 
            "message": "Параметр 'name' обязателен.",
            "used_objects": [],
            "not_used_objects": []
        }), 400

    result = geo.get_point_coords_from_geodb(name)
    
    used_objects = []
    not_used_objects = []
    
    if result.get("status") == "ok":
        used_objects.append({
            "name": name,
            "type": "geographical_entity"
        })
    else:
        not_used_objects.append({
            "name": name, 
            "type": "geographical_entity"
        })
    
    result["used_objects"] = used_objects
    result["not_used_objects"] = not_used_objects
    
    return jsonify(result)

@coordinates_bp.route("/coords_to_map", methods=["POST"])
def api_coords_to_map():
    t0 = time.perf_counter()
    data = request.get_json()
    t_after_parse = time.perf_counter()
    lat = data.get("latitude")
    lon = data.get("longitude")
    radius = data.get("radius_km", 30)
    object_type = data.get("object_type")
    species_name = data.get("species_name")
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist_param = request.args.get("in_stoplist", "1")
    try:
        if in_stoplist_param.lower() in ['false', 'true']:
            in_stoplist = 1
        else:
            in_stoplist = int(in_stoplist_param)
    except (ValueError, TypeError):
        in_stoplist = 1
    
    cache_params = {
        "latitude": lat,
        "longitude": lon,
        "radius_km": radius,
        "object_type": object_type,
        "species_name": species_name,
        "in_stoplist": in_stoplist,
        "version": "v2"
    }
    
    redis_key = f"cache:coords_search:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "search_time": 0,
        "parse_time": round(t_after_parse - t0, 3)
    }

    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)

    logger.debug(f"""Параметры:{data}""")
    if not lat or not lon:
        response = {
            "status": "error", 
            "message": "Не заданы координаты.",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400

    resolved_species_info = None
    if species_name:
        resolved_species_info = search_service.resolve_object_synonym(species_name, "biological_entity")
        
        debug_info["species_resolution"] = {
            "original_name": species_name,
            "resolved_info": resolved_species_info
        }
        
        if resolved_species_info.get("resolved", False):
            species_name = resolved_species_info["main_form"]
            logger.info(f"✅ Разрешен синоним вида: '{resolved_species_info['original_name']}' -> '{species_name}'")
        else:
            logger.info(f"ℹ️ Синоним для вида '{species_name}' не найден, используем оригинальное название")

    t3 = time.perf_counter()
    
    try:
        t1 = time.perf_counter()
        result = search_service.get_nearby_objects(
            latitude=float(lat),
            longitude=float(lon),
            radius_km=float(radius),
            object_type=object_type,
            species_name=species_name,
            in_stoplist=in_stoplist
        )
        t2 = time.perf_counter()
        objects = result.get("objects", [])
        answer = result.get("answer", "")
        
        debug_info["search_time"] = round(t2 - t1, 3)
        debug_info["parameters"] = {
            "latitude": lat,
            "longitude": lon,
            "radius_km": radius,
            "object_type": object_type,
            "species_name": species_name,
            "in_stoplist": in_stoplist
        }
        debug_info["objects_count"] = len(objects)
        debug_info["search_query_details"] = result.get("debug_info", {})
        
        if resolved_species_info:
            debug_info["species_synonym_resolution"] = {
                "original_name": resolved_species_info.get("original_name"),
                "resolved_name": species_name,
                "resolved": resolved_species_info.get("resolved", False)
            }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer,
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)

        safe_objects = []
        stoplisted_objects = []
        
        for obj in objects:
            feature_data = obj.get("features", {})
            obj_in_stoplist = feature_data.get("in_stoplist")
            
            try:
                requested_level = int(in_stoplist)
                if obj_in_stoplist is None or int(obj_in_stoplist) <= requested_level:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
            except (ValueError, TypeError):
                if obj_in_stoplist is None or int(obj_in_stoplist) <= 1:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
                    logger.info(f"Исключен объект с in_stoplist={obj_in_stoplist}: {obj.get('name', 'Без имени')}")
        
        objects = safe_objects
        
        if stoplisted_objects:
            answer = f"{answer} (исключено {len(stoplisted_objects)} объектов по уровню безопасности)"
        
        debug_info["stoplist_filter"] = {
            "total_before_filter": len(result.get("objects", [])),
            "safe_after_filter": len(objects),
            "stoplisted_count": len(stoplisted_objects)
        }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer,
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
                response["in_stoplist_filter_applied"] = True
                response["in_stoplist_level"] = in_stoplist
            return jsonify(response)

        valid_objects = []
        object_details = []
        used_objects = []
        not_used_objects = []
        
        for obj in objects:
            try:
                if obj.get("geojson") and obj["geojson"].get("coordinates"):
                    if isinstance(obj["geojson"]["coordinates"][0], (int, float)):
                        lon, lat = obj["geojson"]["coordinates"]
                        if -180 <= lon <= 180 and -90 <= lat <= 90:
                            valid_objects.append(obj)
                            object_details.append({
                                "id": obj.get("id", "unknown"),
                                "name": obj.get("name", "Без имени"),
                                "type": obj.get("type", "unknown"),
                                "distance_km": obj.get("distance", "unknown")
                            })
                            used_objects.append({
                                "name": obj.get("name", "Без имени"),
                                "type": obj.get("type", "unknown"),
                                "distance_km": obj.get("distance", "unknown"),
                                "geometry_type": "point"
                            })
                    elif isinstance(obj["geojson"]["coordinates"][0], list):
                        first_coord = obj["geojson"]["coordinates"][0][0]
                        if isinstance(first_coord, (int, float)):
                            if -180 <= first_coord <= 180:
                                valid_objects.append(obj)
                                object_details.append({
                                    "id": obj.get("id", "unknown"),
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown")
                                })
                                used_objects.append({
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown"),
                                    "geometry_type": "polygon"
                                })
                        elif len(first_coord) >= 2:
                            lon, lat = first_coord[:2]
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                valid_objects.append(obj)
                                object_details.append({
                                    "id": obj.get("id", "unknown"),
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown")
                                })
                                used_objects.append({
                                    "name": obj.get("name", "Без имени"),
                                    "type": obj.get("type", "unknown"),
                                    "distance_km": obj.get("distance", "unknown"),
                                    "geometry_type": "complex"
                                })
                else:
                    not_used_objects.append({
                        "name": obj.get("name", "Без имени"),
                        "type": obj.get("type", "unknown"),
                        "distance_km": obj.get("distance", "unknown"),
                        "reason": "no_geometry"
                    })
            except Exception as e:
                logger.warning(f"Invalid geometry in object {obj.get('name')}: {str(e)}")
                not_used_objects.append({
                    "name": obj.get("name", "Без имени"),
                    "type": obj.get("type", "unknown"),
                    "distance_km": obj.get("distance", "unknown"),
                    "reason": "invalid_geometry",
                    "error": str(e)
                })
                continue

        debug_info["valid_objects_count"] = len(valid_objects)
        debug_info["object_details"] = object_details
        debug_info["validation_errors"] = len(objects) - len(valid_objects)

        if not valid_objects:
            response = {
                "status": "error",
                "message": "Найдены объекты, но их координаты недействительны для отображения",
                "used_objects": [],
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
                response["in_stoplist_filter_applied"] = True
                response["in_stoplist_level"] = in_stoplist
            return jsonify(response)

        try:
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(valid_objects, map_name)
            t3 = time.perf_counter()
            map_result["count"] = len(valid_objects)
            map_result["answer"] = answer
            map_result["names"] = [obj.get("name", "Без имени") for obj in valid_objects]
            map_result["used_objects"] = used_objects
            map_result["not_used_objects"] = not_used_objects
            map_result["in_stoplist_filter_applied"] = True
            map_result["in_stoplist_level"] = in_stoplist
            map_result["stoplisted_count"] = len(stoplisted_objects)
            
            if resolved_species_info and resolved_species_info.get("resolved", False):
                map_result["species_synonym_resolution"] = {
                    "original_name": resolved_species_info["original_name"],
                    "resolved_name": species_name,
                    "resolved": True
                }
            
            debug_info["render_time"] = round(t3 - t2, 3)
            debug_info["total_time"] = round(time.perf_counter() - t0, 3)
            debug_info["map_generation"] = {
                "static_map": map_result.get("static_map"),
                "interactive_map": map_result.get("interactive_map"),
                "map_name": map_name
            }
            
            if debug_mode:
                map_result["debug"] = debug_info

            set_cached_result(redis_key, map_result, expire_time=1800)
            return jsonify(map_result)
        except Exception as e:
            logger.error(f"Ошибка отрисовки карты: {e}")
            debug_info["render_error"] = str(e)
            response = {
                "status": "error", 
                "message": f"Ошибка отрисовки карты: {e}",
                "objects": [obj["name"] for obj in valid_objects],
                "answer": answer,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist,
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
            
    except Exception as e:
        logger.error(f"Ошибка поиска рядом: {e}")
        debug_info["search_error"] = str(e)
        response = {
            "status": "error", 
            "message": f"Ошибка поиска рядом: {e}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500
    finally:
        logging.info(
            "coords_to_map timings parse=%.3f search=%.3f render=%.3f total=%.3f",
            t_after_parse - t0,
            t2 - t1,
            t3 - t2,
            time.perf_counter() - t0,
        )