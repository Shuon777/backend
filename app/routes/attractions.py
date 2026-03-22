import json
import logging
import time
from flask import Blueprint, request, jsonify
from app.services import search_service, relational_service, geo
from app.utils import generate_cache_key, get_cached_result, set_cached_result, convert_floats

attractions_bp = Blueprint('attractions', __name__)
logger = logging.getLogger(__name__)

@attractions_bp.route("/find_off_near_attractions", methods=["POST"])
def find_off_near_attractions():
    """
    Поиск ОФФ около достопримечательностей в указанной области
    Args в JSON body:
    - area_name: название области (например, "Иркутск") - опционально
    - area_polygon: GeoJSON полигона области - опционально (если нет area_name)
    - attraction_types: список типов достопримечательностей (например, ["музей", "памятник"])
    - attraction_subtypes: список подтипов достопримечательностей - опционально
    - off_types: типы ОФФ для поиска (например, ["biological_entity"]) - по умолчанию biological_entity
    - buffer_radius_km: радиус буферной зоны вокруг достопримечательностей в км (по умолчанию 1)
    - limit: ограничение на количество результатов
    - in_stoplist: уровень фильтрации безопасности
    """
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    
    data = request.get_json()
    logger.info(f"🔍 /find_off_near_attractions - запрос: {data}")
    
    # Параметры для кеша
    cache_params = {
        "area_name": data.get("area_name"),
        "area_polygon": data.get("area_polygon"),
        "attraction_types": data.get("attraction_types"),
        "attraction_subtypes": data.get("attraction_subtypes"),
        "off_types": data.get("off_types", ["biological_entity"]),
        "buffer_radius_km": data.get("buffer_radius_km", 1),
        "limit": data.get("limit", 50),
        "in_stoplist": data.get("in_stoplist", "1"),
        "version": "v1"
    }
    
    redis_key = f"cache:off_near_attractions:{generate_cache_key(cache_params)}"
    debug_info = {
        "timestamp": time.time(),
        "cache_key": redis_key,
        "steps": []
    }
    
    # Проверяем кеш
    cache_hit, cached_result = get_cached_result(redis_key, debug_info)
    if cache_hit:
        if debug_mode:
            cached_result["debug"] = debug_info
        return jsonify(cached_result)
    
    # Извлекаем параметры
    area_name = data.get("area_name")
    area_polygon = data.get("area_polygon")
    attraction_types = data.get("attraction_types", [])
    attraction_subtypes = data.get("attraction_subtypes", [])
    off_types = data.get("off_types", ["biological_entity"])
    buffer_radius_km = data.get("buffer_radius_km", 1.0)
    limit = data.get("limit", 50)
    in_stoplist = data.get("in_stoplist", "1")
    
    # Валидация параметров
    if not attraction_types:
        response = {
            "error": "Необходимо указать attraction_types (типы достопримечательностей)",
            "used_objects": [],
            "not_used_objects": []
        }
        return jsonify(response), 400
    
    # Debug информация
    debug_info["parameters"] = {
        "area_name": area_name,
        "has_area_polygon": bool(area_polygon),
        "attraction_types": attraction_types,
        "attraction_subtypes": attraction_subtypes,
        "off_types": off_types,
        "buffer_radius_km": buffer_radius_km,
        "limit": limit,
        "in_stoplist": in_stoplist
    }
    
    try:
        # 1. Получаем полигон области (если указан area_name)
        area_geometry = None
        area_info = None
        
        if area_name:
            # Ищем полигон области
            area_result = relational_service.find_area_geometry(area_name)
            if area_result:
                area_geometry = area_result.get("geometry")
                area_info = area_result.get("area_info", {})
                debug_info["steps"].append({
                    "step": "area_search",
                    "found": True,
                    "area_title": area_info.get('title', area_name)
                })
            else:
                debug_info["steps"].append({
                    "step": "area_search",
                    "found": False,
                    "error": f"Area '{area_name}' not found"
                })
        elif area_polygon:
            area_geometry = area_polygon
            debug_info["steps"].append({
                "step": "area_polygon_provided",
                "found": True
            })
        
        # 2. Ищем достопримечательности в области
        attractions = []
        all_attractions = []
        
        for attraction_type in attraction_types:
            search_object_type = "geographical_entity"
            try:
                if area_geometry:
                    results = relational_service.get_objects_in_area_by_type(
                        area_geometry=area_geometry,
                        object_type=search_object_type,
                        object_subtype=attraction_type if attraction_subtypes else None,
                        limit=limit * 2,
                        search_around=False
                    )
                else:
                    results = relational_service.search_objects_by_name(
                        object_name="",
                        object_type=search_object_type,
                        object_subtype=attraction_type if attraction_subtypes else None,
                        limit=limit * 2
                    )
                
                if results:
                    for obj in results:
                        obj["attraction_type"] = attraction_type
                    all_attractions.extend(results)
                    
            except Exception as e:
                logger.error(f"Ошибка поиска достопримечательностей типа {attraction_type}: {str(e)}")
                debug_info["steps"].append({
                    "step": f"attraction_search_{attraction_type}",
                    "error": str(e)
                })
        
        # Фильтруем достопримечательности по подтипам если указаны
        if attraction_subtypes:
            filtered_attractions = []
            for obj in all_attractions:
                features = obj.get("features", {})
                geo_type = features.get("geo_type", {})
                specific_types = geo_type.get("specific_types", [])
                
                if any(subtype in specific_types for subtype in attraction_subtypes):
                    filtered_attractions.append(obj)
            attractions = filtered_attractions[:limit]
        else:
            attractions = all_attractions[:limit]
        
        debug_info["attraction_search"] = {
            "total_found": len(all_attractions),
            "after_filtering": len(attractions),
            "attraction_types_used": list(set([a.get("attraction_type", "unknown") for a in attractions]))
        }
        
        if not attractions:
            response = {
                "status": "no_attractions",
                "message": f"В указанной области не найдено достопримечательностей типов: {attraction_types}",
                "used_objects": [],
                "not_used_objects": []
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        # 3. Ищем ОФФ около каждой достопримечательности
        all_off_results = []
        attraction_off_map = {}
        
        for attraction in attractions:
            attraction_id = attraction.get("id")
            attraction_name = attraction.get("name", "Неизвестная достопримечательность")
            attraction_geojson = attraction.get("geojson")
            
            if not attraction_geojson:
                continue
            
            try:
                buffer_geometry = search_service.geo_service.create_buffer_geometry(
                    attraction_geojson, 
                    buffer_radius_km
                )
                
                if not buffer_geometry:
                    continue
                
                for off_type in off_types:
                    off_results = search_service.geo_service.get_objects_in_polygon(
                        polygon_geojson=buffer_geometry,
                        object_type=off_type,
                        limit=20
                    )
                    
                    if off_results:
                        clipped_results = []
                        for off_obj in off_results:
                            original_geojson = off_obj.get("geojson")
                            
                            try:
                                if original_geojson and original_geojson.get("type") in ["Polygon", "MultiPolygon"]:
                                    from shapely.geometry import shape, mapping
                                    
                                    buffer_shape = shape(buffer_geometry)
                                    off_shape = shape(original_geojson)
                                    
                                    if buffer_shape.intersects(off_shape):
                                        intersection = buffer_shape.intersection(off_shape)
                                        if not intersection.is_empty:
                                            clipped_geojson = mapping(intersection)
                                            off_obj["geojson"] = clipped_geojson
                                        else:
                                            continue
                                    else:
                                        continue
                                elif original_geojson and original_geojson.get("type") == "Point":
                                    buffer_shape = shape(buffer_geometry)
                                    off_shape = shape(original_geojson)
                                    if not buffer_shape.contains(off_shape):
                                        continue
                                    
                            except Exception as e:
                                logger.warning(f"Ошибка обрезки геометрии ОФФ: {str(e)}")
                            
                            off_obj["near_attraction"] = {
                                "id": attraction_id,
                                "name": attraction_name,
                                "type": attraction.get("attraction_type", "unknown"),
                                "distance_km": buffer_radius_km
                            }
                            
                            clipped_results.append(off_obj)
                        
                        if clipped_results:
                            unique_off = {}
                            for off_obj in clipped_results:
                                key = off_obj.get("id") or f"{off_obj.get('name')}_{off_obj.get('geojson', {}).get('coordinates', [])}"
                                if key not in unique_off:
                                    unique_off[key] = off_obj
                            
                            attraction_off_map.setdefault(attraction_id, []).extend(list(unique_off.values()))
                            all_off_results.extend(list(unique_off.values()))
                            
            except Exception as e:
                logger.error(f"Ошибка поиска ОФФ около {attraction_name}: {str(e)}")
                debug_info["steps"].append({
                    "step": f"off_search_near_{attraction_id}",
                    "error": str(e)
                })
        
        # 4. Подготавливаем данные для карты
        map_objects = []
        used_objects = []
        not_used_objects = []
        
        if area_geometry:
            map_objects.append({
                'tooltip': f"Область поиска: {area_info.get('title', area_name) if area_name else 'Пользовательский полигон'}",
                'popup': f"<h6>Область поиска</h6><p>Поиск достопримечательностей в этой области</p>",
                'geojson': area_geometry,
                'style': {'color': 'blue', 'fillOpacity': 0.1, 'weight': 2}
            })
        
        for attraction in attractions:
            attraction_id = attraction.get("id")
            attraction_name = attraction.get("name")
            attraction_type = attraction.get("attraction_type", "unknown")
            geojson = attraction.get("geojson")
            
            used_objects.append({
                "name": attraction_name,
                "type": "attraction",
                "attraction_type": attraction_type,
                "id": attraction_id
            })
            
            off_list = attraction_off_map.get(attraction_id, [])
            off_count = len(off_list)
            
            popup_html = f"<h6>{attraction_name}</h6>"
            popup_html += f"<p>Тип: {attraction_type}</p>"
            
            if off_count > 0:
                popup_html += f"<p>Найдено ОФФ поблизости: {off_count}</p>"
                popup_html += "<ul>"
                for off in off_list[:5]:
                    popup_html += f"<li>{off.get('name', 'Неизвестный ОФФ')}</li>"
                if off_count > 5:
                    popup_html += f"<li>... и еще {off_count - 5}</li>"
                popup_html += "</ul>"
            else:
                popup_html += "<p>ОФФ поблизости не найдены</p>"
            
            map_objects.append({
                'tooltip': f"{attraction_name} ({attraction_type})",
                'popup': popup_html,
                'geojson': geojson,
                'style': {'color': 'red', 'fillOpacity': 0.3, 'weight': 3}
            })
            
            try:
                buffer_geometry = search_service.geo_service.create_buffer_geometry(
                    geojson, 
                    buffer_radius_km
                )
                if buffer_geometry:
                    map_objects.append({
                        'tooltip': f"Зона поиска ОФФ вокруг {attraction_name}",
                        'popup': f"<h6>Буферная зона</h6><p>Зона поиска ОФФ в радиусе {buffer_radius_km} км вокруг {attraction_name}</p>",
                        'geojson': buffer_geometry,
                        'style': {'color': 'orange', 'fillOpacity': 0.1, 'weight': 2}
                    })
            except Exception as e:
                logger.warning(f"Не удалось создать буфер для {attraction_name}: {str(e)}")
        
        off_names = set()
        for off in all_off_results:
            off_name = off.get("name", "Неизвестный ОФФ")
            off_type = off.get("type", "unknown")
            geojson = off.get("geojson")
            near_attraction = off.get("near_attraction", {})
            
            if off_name not in off_names:
                off_names.add(off_name)
                used_objects.append({
                    "name": off_name,
                    "type": off_type,
                    "near_attraction": near_attraction.get("name")
                })
            
            popup_html = f"<h6>{off_name}</h6>"
            popup_html += f"<p>Тип: {off_type}</p>"
            if near_attraction:
                popup_html += f"<p>Найден около: {near_attraction.get('name')} ({near_attraction.get('type')})</p>"
                popup_html += f"<p>Расстояние: до {near_attraction.get('distance_km')} км</p>"
            
            map_objects.append({
                'tooltip': off_name,
                'popup': popup_html,
                'geojson': geojson,
                'style': {'color': 'green', 'fillOpacity': 0.5, 'weight': 2}
            })
        
        if not map_objects:
            response = {
                "status": "no_results",
                "message": "Не найдено объектов для отображения на карте",
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        try:
            map_name = redis_key.replace("cache:", "map_").replace(":", "_")
            map_result = geo.draw_custom_geometries(map_objects, map_name)
            
            response_data = {
                "map": map_result,
                "statistics": {
                    "attractions_found": len(attractions),
                    "off_found": len(all_off_results),
                    "unique_off_names": len(off_names),
                    "attraction_types": list(set([a.get("attraction_type", "unknown") for a in attractions])),
                    "buffer_radius_km": buffer_radius_km
                },
                "used_objects": used_objects,
                "not_used_objects": not_used_objects,
                "in_stoplist_filter_applied": True,
                "in_stoplist_level": in_stoplist
            }
            
            if area_name:
                response_data["answer"] = f"В области '{area_name}' найдено {len(attractions)} достопримечательностей. Около них обнаружено {len(all_off_results)} ОФФ ({len(off_names)} уникальных видов)."
            else:
                response_data["answer"] = f"Найдено {len(attractions)} достопримечательностей. Около них обнаружено {len(all_off_results)} ОФФ ({len(off_names)} уникальных видов)."
            
            response_data["attractions"] = [
                {
                    "id": a.get("id"),
                    "name": a.get("name"),
                    "type": a.get("attraction_type"),
                    "off_count": len(attraction_off_map.get(a.get("id"), []))
                }
                for a in attractions[:10]
            ]
            
            response_data["off_objects"] = [
                {
                    "name": off.get("name"),
                    "type": off.get("type"),
                    "near_attraction": off.get("near_attraction", {}).get("name")
                }
                for off in all_off_results[:20]
            ]
            
            if debug_mode:
                debug_info["results_summary"] = {
                    "total_map_objects": len(map_objects),
                    "attraction_off_distribution": {aid: len(offs) for aid, offs in attraction_off_map.items()}
                }
                response_data["debug"] = debug_info
            
            set_cached_result(redis_key, response_data, expire_time=3600)
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Ошибка создания карты: {str(e)}")
            error_response = {
                "error": f"Ошибка создания карты: {str(e)}",
                "statistics": {
                    "attractions_found": len(attractions),
                    "off_found": len(all_off_results)
                },
                "used_objects": used_objects,
                "not_used_objects": not_used_objects
            }
            if debug_mode:
                error_response["debug"] = debug_info
            return jsonify(error_response), 500
            
    except Exception as e:
        logger.error(f"Ошибка в /find_off_near_attractions: {str(e)}", exc_info=True)
        error_response = {
            "error": f"Внутренняя ошибка сервера: {str(e)}",
            "used_objects": [],
            "not_used_objects": []
        }
        if debug_mode:
            error_response["debug"] = debug_info
        return jsonify(error_response), 500