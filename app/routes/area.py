import logging
import time
from flask import Blueprint, request, jsonify
from app.services import search_service, relational_service, geo
from app.utils import generate_cache_key, get_cached_result, set_cached_result, extract_external_id

area_bp = Blueprint('area', __name__)
logger = logging.getLogger(__name__)

@area_bp.route("/objects_in_area_by_type", methods=["POST"])
def objects_in_area_by_type():
    data = request.get_json()
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"

    cache_params = {
        "area_name": data.get("area_name"),
        "object_type": data.get("object_type", "all"),
        "object_subtype": data.get("object_subtype"),
        "object_name": data.get("object_name"),
        "limit": data.get("limit", 20),
        "search_around": data.get("search_around", False),
        "buffer_radius_km": data.get("buffer_radius_km", 10.0),
        "version": "v2"
    }
    
    redis_key = f"cache:area_search:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "steps": []
    }

    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)

    logger.info(f"📦 /objects_in_area_by_type - GET params: {dict(request.args)}")
    logger.info(f"📦 /objects_in_area_by_type - POST data: {request.get_json()}")

    area_name = data.get("area_name")
    object_type = data.get("object_type", "all") 
    object_subtype = data.get("object_subtype")
    object_name = data.get("object_name")
    limit = data.get("limit", 1500)
    search_around = data.get("search_around", False)
    buffer_radius_km = data.get("buffer_radius_km", 10.0)

    debug_info["parameters"] = {
        "area_name": area_name,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "object_name": object_name,
        "limit": limit,
        "search_around": search_around,
        "buffer_radius_km": buffer_radius_km
    }
    
    resolved_object_info = None
    if object_name:
        resolved_object_info = search_service.resolve_object_synonym(object_name, object_type)
        
        debug_info["synonym_resolution"] = {
            "original_name": object_name,
            "original_type": object_type,
            "resolved_info": resolved_object_info
        }
        
        if resolved_object_info.get("resolved", False):
            object_name = resolved_object_info["main_form"]
            if object_type != "all":
                object_type = resolved_object_info["object_type"]
            logger.info(f"✅ Разрешен синоним объекта: '{resolved_object_info['original_name']}' -> '{object_name}' (тип: {object_type})")
        else:
            logger.info(f"ℹ️ Синоним для объекта '{object_name}' не найден, используем оригинальное название")
    
    if not area_name and object_name:
        debug_info["steps"].append({
            "step": "direct_object_search",
            "reason": "area_name not provided, searching object directly",
            "resolved_name": object_name,
            "resolved_type": object_type
        })
        
        try:
            results = search_service.search_objects_directly_by_name(
                object_name=object_name,
                object_type=object_type,
                object_subtype=object_subtype,
                limit=limit
            )
            
            objects = results.get("objects", [])
            answer = results.get("answer", "")
            
            debug_info["search_results"] = {
                "total_objects_found": len(objects),
                "search_type": "direct_object_search"
            }
            
            if not objects:
                response = {
                    "status": "no_objects", 
                    "message": answer
                }
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response)
            
            objects_for_map = []
            used_objects = []
            
            for obj in objects:
                name = obj.get('name', 'Без имени')
                description = obj.get('description', '')
                geojson = obj.get('geojson', {})
                obj_type = obj.get('type', 'unknown')
                
                used_objects.append({
                    "name": name,
                    "type": obj_type,
                    "external_id": extract_external_id(obj.get('features', {})),
                    "geometry_type": obj.get('geometry_type')
                })
                
                popup_html = f"<h6>{name}</h6>"
                if description:
                    short_desc = description[:200] + "..." if len(description) > 200 else description
                    popup_html += f"<p>{short_desc}</p>"
                
                objects_for_map.append({
                    'tooltip': name,
                    'popup': popup_html,
                    'geojson': geojson
                })
            
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(objects_for_map, map_name)
            
            detailed_objects = []
            for obj in objects:
                features = obj.get('features', {})
                external_id = extract_external_id(features)
                
                detailed_objects.append({
                    "name": obj.get('name'), 
                    "description": obj.get('description'),
                    "type": obj.get('type'),
                    "external_id": external_id,
                    "geometry_type": obj.get('geometry_type'),
                    "primary_types": obj.get('primary_types', []),
                    "specific_types": obj.get('specific_types', [])
                })
            
            map_result["count"] = len(objects)
            map_result["answer"] = answer
            map_result["objects"] = detailed_objects
            map_result["used_objects"] = used_objects
            map_result["not_used_objects"] = []
            
            if resolved_object_info and resolved_object_info.get("resolved", False):
                map_result["synonym_resolution"] = {
                    "original_name": resolved_object_info["original_name"],
                    "resolved_name": object_name,
                    "original_type": resolved_object_info.get("original_type", object_type)
                }
            
            if debug_mode:
                objects_with_external_id = [obj for obj in detailed_objects if obj.get('external_id')]
                debug_info["external_id_stats"] = {
                    "total_objects": len(detailed_objects),
                    "with_external_id": len(objects_with_external_id)
                }
                debug_info["visualization"] = {
                    "map_name": map_name,
                    "total_objects_on_map": len(objects_for_map),
                    "search_type": "direct_object_search"
                }
                map_result["debug"] = debug_info

            set_cached_result(redis_key, map_result, expire_time=1800)
            return jsonify(map_result)
            
        except Exception as e:
            logger.error(f"Ошибка прямого поиска объекта: {str(e)}")
            debug_info["error"] = str(e)
            response = {"error": "Внутренняя ошибка сервера при поиске объекта"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
    
    if not area_name:
        response = {"error": "area_name is required when no object_name provided"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    
    area_geometry = None
    area_info = None
    
    try:
        area_results = relational_service.find_area_geometry(area_name)
        
        if area_results:
            area_geometry = area_results.get("geometry")
            area_info = area_results.get("area_info", {})
            
            debug_info["steps"].append({
                "step": "area_search",
                "found": True,
                "area_title": area_info.get('title', area_name),
                "source": area_info.get('source', 'unknown')
            })
        else:
            debug_info["steps"].append({
                "step": "area_search", 
                "found": False,
                "error": "Area not found in map_content"
            })
            
    except Exception as e:
        logger.error(f"Ошибка поиска области через relational_service: {str(e)}")
        debug_info["steps"].append({
            "step": "area_search",
            "error": str(e)
        })
    
    if not area_geometry:
        response = {"error": f"Полигон для области '{area_name}' не найден в базе данных"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 404

    try:
        results = search_service.get_objects_in_area_by_type(
            area_geometry=area_geometry,
            object_type=object_type,
            object_subtype=object_subtype,
            object_name=object_name,
            limit=int(limit),
            search_around=search_around,
            buffer_radius_km=float(buffer_radius_km)
        )
        
        objects = results.get("objects", [])
        answer = results.get("answer", "")
        search_stats = results.get("search_stats", {})
        
        debug_info["search_results"] = {
            "total_objects_found": len(objects),
            "search_criteria": {
                "object_type": object_type,
                "object_subtype": object_subtype,
                "object_name": object_name,
                "search_around": search_around,
                "buffer_radius_km": buffer_radius_km
            },
            "location_stats": search_stats
        }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        objects_for_map = []
        used_objects = []
        
        area_title = area_info.get('title', area_name) if area_info else area_name
        objects_for_map.append({
            'tooltip': f"Область поиска: {area_title}",
            'popup': f"<h6>{area_title}</h6><p>Область поиска</p>",
            'geojson': area_geometry
        })
        
        buffer_geometry = None
        if search_around:
            buffer_geometry = search_service.geo_service.create_buffer_geometry(area_geometry, buffer_radius_km)
            if buffer_geometry:
                objects_for_map.append({
                    'tooltip': f"Зона поиска (+{buffer_radius_km} км)",
                    'popup': f"<h6>Зона поиска</h6><p>Буферная зона {buffer_radius_km} км вокруг области</p>",
                    'geojson': buffer_geometry,
                    'style': {'color': 'orange', 'fillOpacity': 0.1, 'weight': 2}
                })
                debug_info["steps"].append({
                    "step": "buffer_zone_creation",
                    "success": True,
                    "buffer_radius_km": buffer_radius_km
                })
            else:
                debug_info["steps"].append({
                    "step": "buffer_zone_creation", 
                    "success": False,
                    "error": "Failed to create buffer geometry"
                })
        
        for obj in objects:
            name = obj.get('name', 'Без имени')
            description = obj.get('description', '')
            geojson = obj.get('geojson', {})
            location_type = obj.get('location_type', 'inside')
            
            used_objects.append({
                "name": name,
                "type": obj.get('type', 'unknown'),
                "external_id": extract_external_id(obj.get('features', {})),
                "geometry_type": obj.get('geometry_type'),
                "location_type": location_type
            })
            
            popup_html = f"<h6>{name}</h6>"
            if description:
                short_desc = description[:200] + "..." if len(description) > 200 else description
                popup_html += f"<p>{short_desc}</p>"
            
            objects_for_map.append({
                'tooltip': name,
                'popup': popup_html,
                'geojson': geojson
            })
        
        map_name = redis_key.replace("cache:", "map_").replace(":", "_")
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        detailed_objects = []
        for obj in objects:
            features = obj.get('features', {})
            external_id = extract_external_id(features)
            
            detailed_objects.append({
                "name": obj.get('name'), 
                "description": obj.get('description'),
                "type": obj.get('type'),
                "external_id": external_id,
                "geometry_type": obj.get('geometry_type'),
                "primary_types": obj.get('primary_types', []),
                "specific_types": obj.get('specific_types', []),
                "location_type": obj.get('location_type', 'inside')
            })
        
        map_result["count"] = len(objects)
        map_result["answer"] = answer
        map_result["objects"] = detailed_objects
        map_result["search_stats"] = search_stats
        map_result["used_objects"] = used_objects
        map_result["not_used_objects"] = []
        
        if buffer_geometry:
            map_result["buffer_zone"] = {
                "radius_km": buffer_radius_km,
                "geometry": buffer_geometry
            }
        
        if resolved_object_info and resolved_object_info.get("resolved", False):
            map_result["synonym_resolution"] = {
                "original_name": resolved_object_info["original_name"],
                "resolved_name": object_name,
                "original_type": resolved_object_info.get("original_type", object_type)
            }
        
        if debug_mode:
            objects_with_external_id = [obj for obj in detailed_objects if obj.get('external_id')]
            debug_info["external_id_stats"] = {
                "total_objects": len(detailed_objects),
                "with_external_id": len(objects_with_external_id)
            }
            debug_info["visualization"] = {
                "map_name": map_name,
                "total_objects_on_map": len(objects_for_map),
                "area_included": True,
                "buffer_zone_included": search_around and buffer_geometry is not None,
                "objects_inside": search_stats.get('inside_area', 0),
                "objects_around": search_stats.get('around_area', 0)
            }
            map_result["debug"] = debug_info

        set_cached_result(redis_key, map_result, expire_time=3600)
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка поиска объектов по типу в области: {str(e)}")
        debug_info["error"] = str(e)
        response = {"error": "Внутренняя ошибка сервера при поиске"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500