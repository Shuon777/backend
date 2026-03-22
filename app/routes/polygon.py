import json
import logging
import time
from flask import Blueprint, request, jsonify
from shapely.geometry import shape
from app.services import search_service, relational_service, geo
from app.utils import generate_cache_key, get_cached_result, set_cached_result, convert_floats

polygon_bp = Blueprint('polygon', __name__)
logger = logging.getLogger(__name__)

@polygon_bp.route("/objects_in_polygon_simply", methods=["POST"])
def objects_in_polygon_simply():
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    in_stoplist = request.args.get("in_stoplist", "1")
    logger.info(f"📦 /objects_in_polygon_simply - GET params: {dict(request.args)}")
    logger.info(f"📦 /objects_in_polygon_simply - POST data: {request.get_json()}")

    data = request.get_json()
    name = data.get("name")
    buffer_radius_km = data.get("buffer_radius_km", 0)
    object_type = data.get("object_type")
    object_subtype = data.get("object_subtype")
    limit = data.get("limit", 1500)

    cache_params = {
        "name": name,
        "buffer_radius_km": buffer_radius_km,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "limit": limit,
        "in_stoplist": in_stoplist,
        "version": "v2"
    }
    redis_key = f"cache:polygon_simply:{generate_cache_key(cache_params)}"
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

    debug_info["parameters"] = {
        "name": name,
        "buffer_radius_km": buffer_radius_km,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "limit": limit,
        "in_stoplist": in_stoplist
    }
    
    try:
        resolved_info = search_service.resolve_object_synonym(name, "all")
        if resolved_info.get("resolved", False):
            canonical_name = resolved_info["main_form"]
            logger.debug(f"Найден синоним: '{name}' -> '{canonical_name}' (тип: {resolved_info['object_type']})")
            name = canonical_name
            debug_info["steps"].append({
                "step": "universal_synonym_resolution", 
                "original_name": data.get("name"),
                "canonical_name": canonical_name,
                "object_type": resolved_info["object_type"],
                "resolved_info": resolved_info
            })
    except Exception as e:
        logger.warning(f"Ошибка при проверке синонимов для '{name}': {e}")
        debug_info["steps"].append({
            "step": "synonym_resolution",
            "error": str(e)
        })
    
    entry = relational_service.find_geometry(name)
    if not entry or "geometry" not in entry:
        from infrastructure.geo_db_store import find_place_flexible
        flexible_result = find_place_flexible(name)
        if flexible_result and flexible_result.get("status") == "found":
            entry = flexible_result["record"]
            logger.debug(f"Найдено через гибкий поиск: '{name}' -> '{flexible_result['name']}'")
            debug_info["steps"].append({
                "step": "flexible_search",
                "found_name": flexible_result['name'],
                "original_name": name
            })
        else:
            logger.debug(f"Геометрия для '{name}' не найдена")
            response = {"error": f"Геометрия для '{name}' не найдена"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404
    
    polygon = entry["geometry"]
    debug_info["geometry_source"] = {
        "source": "database" if entry else "flexible_search",
        "entry_id": entry.get("id", "unknown") if entry else "unknown"
    }

    if not polygon:
        response = {"error": "Polygon not specified"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    
    try:
        results = search_service.get_objects_in_polygon(
            polygon_geojson=polygon,
            buffer_radius_km=float(buffer_radius_km),
            object_type=object_type,
            object_subtype=object_subtype,
            limit=int(limit)
        )

        objects = results.get("objects", [])
        answer = results.get("answer", "")
        
        count_raw_from_db = len(objects)
        
        total_objects_before = len(objects)
        biological_objects_before = [obj for obj in objects if obj.get('type') in ['Объект флоры','Объект фауны']]
        biological_names_before = list(set(obj.get('name', 'Без имени') for obj in biological_objects_before))
        
        debug_info["before_filter"] = {
            "total_objects": total_objects_before,
            "biological_objects_count": len(biological_objects_before),
            "biological_names": biological_names_before
        }
        
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
            except (ValueError, TypeError):
                if obj_in_stoplist is None or int(obj_in_stoplist) <= 1:
                    safe_objects.append(obj)
                else:
                    stoplisted_objects.append(obj)
        
        objects = safe_objects
        count_safe_after_filter = len(objects)

        total_objects_after = len(objects)
        biological_objects_after = [obj for obj in objects if obj.get('type') in ['Объект флоры','Объект фауны']]
        biological_names_after = list(set(obj.get('name', 'Без имени') for obj in biological_objects_after))
        
        all_biological_names = sorted(biological_names_after)
        
        if stoplisted_objects:
            answer = f"{answer} (исключено {len(stoplisted_objects)} объектов по уровню безопасности)"
        
        debug_info["search_results"] = {
            "total_objects": len(objects),
            "object_types": {},
            "polygon_area": "calculated" if polygon else "unknown"
        }
        
        debug_info["stoplist_filter"] = {
            "total_before_filter": total_objects_before,
            "safe_after_filter": total_objects_after,
            "stoplisted_count": len(stoplisted_objects),
            "biological_before_filter": len(biological_objects_before),
            "biological_after_filter": len(biological_objects_after),
            "biological_names_before": biological_names_before,
            "biological_names_after": biological_names_after
        }
        
        for obj in objects:
            obj_type = obj.get("type", "unknown")
            if obj_type not in debug_info["search_results"]["object_types"]:
                debug_info["search_results"]["object_types"][obj_type] = 0
            debug_info["search_results"]["object_types"][obj_type] += 1
            
    except ValueError:
        response = {"error": "Invalid parameters format"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    except Exception as e:
        logger.error(f"Ошибка при поиске объектов в полигоне: {e}")
        debug_info["search_error"] = str(e)
        response = {"error": "Внутренняя ошибка сервера при поиске"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500

    if not objects:
        response = {
            "status": "no_objects", 
            "message": answer,
            "used_objects": [],
            "not_used_objects": [],
            "all_biological_names": []
        }
        if debug_mode:
            response["debug"] = debug_info
            response["in_stoplist_filter_applied"] = True
            response["in_stoplist_level"] = in_stoplist
        return jsonify(response)

    grouped_by_geojson = {}
    count_missing_geo = 0
    count_duplicates_in_point = 0
    count_total_in_popups = 0
    
    for obj in objects:
        if 'geojson' not in obj or not obj['geojson']:
            count_missing_geo += 1
            continue
        geojson_key = json.dumps(obj['geojson'], sort_keys=True)
        obj_type = obj.get('type', 'unknown')
        location_name = obj.get('location_name') or obj.get('name') or 'Неизвестное место'
        
        if geojson_key not in grouped_by_geojson:
            grouped_by_geojson[geojson_key] = {
                'geojson': obj['geojson'],
                'type': obj_type,
                'location_name': location_name,
                'biological_names': []
            }
        
        object_name = obj.get('name', 'Без имени')
        
        if object_name not in grouped_by_geojson[geojson_key]['biological_names']:
            grouped_by_geojson[geojson_key]['biological_names'].append(object_name)
            count_total_in_popups += 1
        else:
            count_duplicates_in_point += 1

    logger.debug(
        f"📊 STATISTICS: "
        f"RawDB={count_raw_from_db} | "
        f"Filtered={count_safe_after_filter} | "
        f"NoGeo={count_missing_geo} | "
        f"Dupes(Hidden)={count_duplicates_in_point} | "
        f"VisibleInPopups={count_total_in_popups} | "
        f"UniqueMapMarkers={len(grouped_by_geojson)}"
    )
    
    objects_for_map = []
    MAX_NAMES_IN_TOOLTIP = 3
    MAX_POPUP_HEIGHT = "300px"
    used_objects = []
    not_used_objects = []

    for group_data in grouped_by_geojson.values():
        biological_names = sorted(group_data['biological_names'])
        location_name = group_data.get('location_name', 'Неизвестное место')
        obj_type = group_data.get('type', 'unknown')
        
        used_objects.append({
            "name": location_name,
            "type": obj_type
        })
        
        if len(biological_names) > MAX_NAMES_IN_TOOLTIP:
            tooltip_text = f"{location_name}: {len(biological_names)} видов"
        else:
            tooltip_text = f"{location_name}: {', '.join(biological_names)}"

        popup_html = f"""
        <div style="max-width: 320px; font-family: Arial, sans-serif;">
            <h5 style="
                margin: 0 0 12px 0; 
                padding: 0; 
                color: #2c3e50; 
                border-bottom: 2px solid #3498db; 
                padding-bottom: 8px;
                font-size: 16px;
            ">{location_name}</h5>
        """

        if obj_type == "biological_entity":
            popup_html += f'''
            <div style="
                font-size: 13px; 
                color: #7f8c8d; 
                margin-bottom: 12px;
                padding: 5px;
                background: #ecf0f1;
                border-radius: 4px;
            ">
                🐾 Обнаружено видов: <strong>{len(biological_names)}</strong>
            </div>
            '''
        else:
            popup_html += f'''
            <div style="
                font-size: 13px; 
                color: #7f8c8d; 
                margin-bottom: 12px;
                padding: 5px;
                background: #ecf0f1;
                border-radius: 4px;
            ">
                📍 Обнаружено объектов: <strong>{len(biological_names)}</strong>
            </div>
            '''

        popup_html += f'''
        <div style="
            max-height: {MAX_POPUP_HEIGHT};
            overflow-y: auto;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 8px;
            background: #f8f9fa;
        ">
            <ul style="
                list-style: none;
                padding: 0;
                margin: 0;
            ">
        '''

        for i, biological_name in enumerate(biological_names):
            bg_color = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            
            popup_html += f'''
            <li style="
                padding: 8px 10px;
                margin: 3px 0;
                background: {bg_color};
                border-left: 4px solid #3498db;
                border-radius: 4px;
                font-size: 13px;
                transition: all 0.2s ease;
            ">{biological_name}</li>
            '''

        popup_html += "</ul></div>"
        popup_html += """
        <style>
            div::-webkit-scrollbar {
                width: 8px;
            }
            div::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            div::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 4px;
            }
            div::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
            li:hover {
                background: #e3f2fd !important;
                transform: translateX(2px);
            }
        </style>
        </div>
        """
        
        objects_for_map.append({
            'tooltip': tooltip_text,
            'popup': popup_html,
            'geojson': group_data['geojson']
        })

    try:
        def get_sort_key(item):
            try:
                geom = shape(item['geojson'])
                area = geom.area
                if area == 0 or geom.geom_type in ['Point', 'MultiPoint', 'LineString', 'MultiLineString']:
                    return -1 
                return area
            except:
                return 0

        objects_for_map.sort(key=get_sort_key, reverse=True)
        logger.debug(f"📐 Objects sorted by area for correct Z-indexing. Count: {len(objects_for_map)}")
    except Exception as e:
        logger.warning(f"Ошибка при сортировке геометрий: {e}")

    try:
        map_name = redis_key.replace("cache:", "map_").replace(":", "_")
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        map_result["count"] = count_safe_after_filter
        map_result["answer"] = answer
        map_result["grouped_names"] = [obj.get("tooltip", "") for obj in objects_for_map]
        map_result["all_biological_names"] = all_biological_names
        map_result["used_objects"] = used_objects
        map_result["not_used_objects"] = not_used_objects
        map_result["in_stoplist_filter_applied"] = True
        map_result["in_stoplist_level"] = in_stoplist
        map_result["stoplisted_count"] = len(stoplisted_objects)
        
        if debug_mode:
            debug_info["visualization"] = {
                "map_name": map_name,
                "objects_count": len(objects_for_map),
                "biological_names_count": len(all_biological_names),
                "biological_names_list": all_biological_names,
                "popup_style": "custom_scrollbar_v2",
                "stats_counters": {
                    "raw_db": count_raw_from_db,
                    "safe_filtered": count_safe_after_filter,
                    "no_geo": count_missing_geo,
                    "duplicates_in_point": count_duplicates_in_point,
                    "visible_in_popups": count_total_in_popups
                }
            }
            map_result["debug"] = debug_info

        set_cached_result(redis_key, map_result, expire_time=2700)
        
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка отрисовки карты: {e}", exc_info=True)
        debug_info["visualization_error"] = str(e)
        response = {
            "status": "error", 
            "message": f"Ошибка отрисовки карты: {e}",
            "used_objects": [],
            "not_used_objects": [],
            "all_biological_names": all_biological_names
        }
        if debug_mode:
            response["debug"] = debug_info
            response["in_stoplist_filter_applied"] = True
            response["in_stoplist_level"] = in_stoplist
        return jsonify(response), 500