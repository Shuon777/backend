from http.client import HTTPException
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import unquote
import os
from infrastructure.geo_db_store import get_place, find_place_flexible
from infrastructure.maps_store import get_map_links
from core.search_service import SearchService
from shapely.geometry import shape
from core.coordinates_finder import GeoProcessor
from infrastructure.db_utils_for_search import Slot_validator
from infrastructure.to_nomn import to_prepositional_phrase, find_place_key
import logging
import time
from core.relational_service import RelationalService
from embedding_config import embedding_config
import json

app = Flask(__name__)
CORS(app)

MAPS_DIR = "/var/www/map_bot/maps"
DOMAIN = "https://testecobot.ru"
geo = GeoProcessor(maps_dir=MAPS_DIR, domain=DOMAIN)
slot_val = Slot_validator()

faiss_index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index_path")
current_model, current_model_path = embedding_config.get_active_model()
embedding_model_path = current_model_path

species_synonyms_path = os.getenv("SPECIES_SYNONYMS_PATH", 
                                 str(Path(__file__).parent / "json_files" / "species_synonyms.json"))

search_service = SearchService(
    faiss_index_path=faiss_index_path,
    embedding_model_path=embedding_model_path,
    species_synonyms_path=species_synonyms_path
)
relational_service = RelationalService(species_synonyms_path=species_synonyms_path)

user_locations = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route("/find_place", methods=["POST"])
def find_place():
    data = request.get_json()
    place = data.get("place")
    if isinstance(place, list):
        return jsonify({"error": "Используйте /find_places для поиска нескольких мест"}), 400
    if not place:
        return jsonify({"error": "Place not specified"}), 400

    record = get_place(place)
    map_links = get_map_links(place)
    if record and map_links:
        return jsonify({
            "static_map": map_links.get("static"),
            "interactive_map": map_links.get("interactive"),
            "answer": f"Место '{place}' уже есть в базе. Отдаем сохраненные карты."
        })

    features = geo.fetch_and_draw(place, True)
    if not features:
        return jsonify({"error": "Place not found", "answer": f"Место '{place}' не найдено."}), 404

    map_links = get_map_links(place)
    return jsonify({
        "static_map": map_links.get("static"),
        "interactive_map": map_links.get("interactive"),
        "answer": f"Найдена и сохранена карта для '{place}'."
    })

@app.route("/find_places", methods=["POST"])
def find_places():
    data = request.get_json()
    places = data.get("places")
    if not places or not isinstance(places, list):
        return jsonify({"error": "Places must be a list of names"}), 400

    result = geo.fetch_and_draw_multiple(places)
    if result["status"] == "ok":
        return jsonify(
            {
                "static_map": result["map_image"],
                "interactive_map": result["web_app_url"],
                "answer": result["answer"]
            }
        )
    else:
        return jsonify({"answer": result["answer"]}), 200

@app.route("/get_species_area", methods=["POST"])
def get_species_area():
    data = request.get_json()
    center = data.get("center")
    region = data.get("region")

    if not center or not region:
        return jsonify({"error": "center and region are required"}), 400

    result = geo.get_species_area_near_center(center, region)

    if result["status"] == "ok":
        return jsonify(
            {
                "static_map": result["map_image"],
                "interactive_map": result["web_app_url"],
                "answer": result["answer"]
            }
        )
    else:
        return jsonify({"answer": result["answer"]}), 200
    
@app.route("/objects_in_polygon", methods=["POST"])
def objects_in_polygon():
    data = request.get_json()
    polygon = data.get("polygon")
    buffer_radius_km = data.get("buffer_radius_km", 0)
    object_type = data.get("object_type")
    limit = data.get("limit", 20)
    
    if not polygon:
        return jsonify({"error": "Polygon not specified"}), 400
    
    try:
        if not validate_geojson_polygon(polygon):
            return jsonify({"error": "Invalid polygon format"}), 400
            
        results = search_service.get_objects_in_polygon(
            polygon_geojson=polygon,
            buffer_radius_km=float(buffer_radius_km),
            object_type=object_type,
            limit=int(limit)
        )
        
        # Добавляем информацию о полигоне и биологических объектах в ответ
        return jsonify({
            "answer": results.get("answer", ""),
            "objects": results.get("objects", []),
            "polygon": results.get("polygon", {}),
            "biological_objects": results.get("biological_objects", "")
        })
        
    except ValueError:
        return jsonify({"error": "Invalid parameters format"}), 400
    except Exception as e:
        logger.error(f"Error in objects_in_polygon: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/objects_in_polygon_simply", methods=["POST"])
def objects_in_polygon_simply():
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    debug_info = {
        "timestamp": time.time(),
        "steps": []
    }
    
    data = request.get_json()
    name = data.get("name")
    buffer_radius_km = data.get("buffer_radius_km", 0)
    object_type = data.get("object_type")
    limit = data.get("limit", 20)
    
    # Debug информация о параметрах
    debug_info["parameters"] = {
        "name": name,
        "buffer_radius_km": buffer_radius_km,
        "object_type": object_type,
        "limit": limit
    }
    
    # Проверяем синонимы перед поиском геометрии
    try:
        # Получаем основное название из синонимов
        synonyms_data = search_service.get_synonyms_for_name(name)
        
        # Если найдены синонимы, используем основное название
        if "error" not in synonyms_data:
            main_names = list(synonyms_data.keys())
            if main_names:
                # Используем первое основное название
                canonical_name = main_names[0]
                logger.debug(f"Найден синоним: '{name}' -> '{canonical_name}'")
                name = canonical_name
                debug_info["steps"].append({
                    "step": "synonym_resolution",
                    "original_name": data.get("name"),
                    "canonical_name": canonical_name,
                    "synonyms_data": synonyms_data
                })
    except Exception as e:
        logger.warning(f"Ошибка при проверке синонимов для '{name}': {e}")
        debug_info["steps"].append({
            "step": "synonym_resolution",
            "error": str(e)
        })
        # Продолжаем с оригинальным именем
    
    entry = get_place(name)
    if not entry or "geometry" not in entry:
        # Если не нашли по основному названию, пробуем найти через гибкий поиск
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
            limit=int(limit)
        )
        objects = results.get("objects", [])
        answer = results.get("answer", "")
        
        # Debug информация о результатах поиска
        debug_info["search_results"] = {
            "total_objects": len(objects),
            "object_types": {},
            "polygon_area": "calculated" if polygon else "unknown"
        }
        
        # Статистика по типам объектов
        for obj in objects:
            obj_type = obj.get("type", "unknown")
            if obj_type not in debug_info["search_results"]["object_types"]:
                debug_info["search_results"]["object_types"][obj_type] = 0
            debug_info["search_results"]["object_types"][obj_type] += 1
            
            # Добавляем ID объектов для отладки
            if "id" in obj:
                if "object_ids" not in debug_info["search_results"]:
                    debug_info["search_results"]["object_ids"] = {}
                if obj_type not in debug_info["search_results"]["object_ids"]:
                    debug_info["search_results"]["object_ids"][obj_type] = []
                debug_info["search_results"]["object_ids"][obj_type].append(obj["id"])
                
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
        response = {"status": "no_objects", "message": answer}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response)

    # Собираем полный список имен для ответа
    all_biological_names = sorted(list(set(
        obj.get('name', 'Без имени') 
        for obj in objects if obj.get('type') == 'biological_entity'
    )))

    # Группируем объекты по геометрии
    grouped_by_geojson = {}
    for obj in objects:
        if 'geojson' not in obj or not obj['geojson']:
            continue
        geojson_key = json.dumps(obj['geojson'], sort_keys=True)
        if geojson_key not in grouped_by_geojson:
            grouped_by_geojson[geojson_key] = {
                'geojson': obj['geojson'],
                'names': []
            }
        object_name = obj.get('name', 'Без имени')
        if object_name not in grouped_by_geojson[geojson_key]['names']:
             grouped_by_geojson[geojson_key]['names'].append(object_name)

    # Debug информация о группировке
    debug_info["grouping"] = {
        "total_groups": len(grouped_by_geojson),
        "objects_per_group": [len(group['names']) for group in grouped_by_geojson.values()]
    }

    # --- НОВАЯ ЛОГИКА ФОРМИРОВАНИЯ ТЕКСТА ---
    objects_for_map = []
    MAX_NAMES_IN_TOOLTIP = 3  # Максимум имен для краткого отображения

    for group_data in grouped_by_geojson.values():
        names = sorted(group_data['names'])
        
        # 1. Создаем краткий текст для Tooltip (при наведении)
        if len(names) > MAX_NAMES_IN_TOOLTIP:
            tooltip_text = f"{', '.join(names[:MAX_NAMES_IN_TOOLTIP])} и еще {len(names) - MAX_NAMES_IN_TOOLTIP}..."
        else:
            tooltip_text = ", ".join(names)

        # 2. Создаем красивый HTML для Popup (при клике)
        popup_html = f"<h6>Обнаружено видов: {len(names)}</h6>"
        popup_html += '<ul style="padding-left: 20px; margin-top: 5px;">'
        for n in names:
            popup_html += f"<li>{n}</li>"
        popup_html += "</ul>"
        
        objects_for_map.append({
            'tooltip': tooltip_text,
            'popup': popup_html,
            'geojson': group_data['geojson']
        })

    try:
        # Используем уникальное имя для файла карты, чтобы избежать проблем с кешированием
        map_name = f"poly_search_{name.replace(' ', '_')}_{int(time.time())}"
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        map_result["count"] = len(objects_for_map)
        map_result["answer"] = answer
        # В 'grouped_names' теперь отправляем краткие имена для tooltip
        map_result["grouped_names"] = [obj.get("tooltip", "") for obj in objects_for_map]
        map_result["all_biological_names"] = all_biological_names
        
        # Добавляем debug информацию
        if debug_mode:
            debug_info["visualization"] = {
                "map_name": map_name,
                "objects_count": len(objects_for_map),
                "biological_names_count": len(all_biological_names)
            }
            map_result["debug"] = debug_info
        
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка отрисовки карты: {e}", exc_info=True)
        debug_info["visualization_error"] = str(e)
        response = {"status": "error", "message": f"Ошибка отрисовки карты: {e}"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500
    
def validate_geojson_polygon(geojson: dict) -> bool:
    """Проверяет, что GeoJSON содержит валидный полигон"""
    try:
        if geojson.get("type") != "Polygon":
            return False
            
        coordinates = geojson.get("coordinates")
        if not coordinates or not isinstance(coordinates, list):
            return False
            
        for ring in coordinates:
            if len(ring) < 4 or ring[0] != ring[-1]:
                return False
                
        return True
    except:
        return False
    
@app.route("/nearby_objects", methods=["POST"])
def nearby_objects():
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    object_type = data.get("object_type")
    radius_km = data.get("radius_km", 10)  # Значение по умолчанию 10 км
    
    if not latitude or not longitude:
        return jsonify({"error": "Missing latitude or longitude"}), 400
    
    try:
        results = search_service.get_nearby_objects(
            latitude=float(latitude),
            longitude=float(longitude),
            object_type=object_type,
            radius_km=float(radius_km)
        )
        return jsonify(results)
    except ValueError:
        return jsonify({"error": "Invalid coordinates format"}), 400
    except Exception as e:
        logger.error(f"Error in nearby_objects: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_radius_intersection", methods=["POST"])
def get_radius_intersection():
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    radius_km = data.get("radius_km", 10)  
    
    if not latitude or not longitude:
        return jsonify({"error": "Missing latitude or longitude"}), 400
    
    try:
        result = search_service.geo_service.get_radius_intersection(
            latitude=float(latitude),
            longitude=float(longitude),
            radius_km=float(radius_km)  
        )

        if result:
            return jsonify({
                "geojson": result,
                "answer": f"Пересечение радиуса {radius_km} км с полигонами успешно вычислено"
            })
        else:
            return jsonify({
                "answer": f"Не найдено полигонов в радиусе {radius_km} км"
            }), 404
            
    except ValueError:
        return jsonify({"error": "Invalid coordinates format"}), 400
    except Exception as e:
        logger.error(f"Error in get_radius_intersection: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/get_radius_intersection_simply", methods=["POST"])
def get_radius_intersection_simply():
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    radius_km = data.get("radius_km", 10)  # Значение по умолчанию 10 км
    
    if not latitude or not longitude:
        return jsonify({"error": "Missing latitude or longitude"}), 400
    
    try:
        result = search_service.geo_service.get_radius_intersection(
            latitude=float(latitude),
            longitude=float(longitude),
            radius_km=float(radius_km)  
        )
        
        if not result:
            return jsonify({
                "answer": f"Не найдено полигонов в радиусе {radius_km} км"
            }), 404

        intersection_geom = result.get("intersection")
        if intersection_geom:
            objects = [{"geojson": intersection_geom}]
            try:
                map_result = geo.draw_custom_geometries(objects, "custom_map_v3")
                
                regions = result.get("regions", [])
                if regions:
                    region_names = [r["name"] for r in regions]
                    map_result["regions"] = regions
                    map_result["answer"] = (
                        f"Найдено пересечение с {len(regions)} регионами: " 
                        + ", ".join(region_names)
                    )
                else:
                    map_result["answer"] = "Найдено пересечение, но регионы не определены"
                    
                return jsonify(map_result)
            except Exception as e:
                return jsonify({
                    "status": "error", 
                    "message": f"Ошибка отрисовки карты: {e}"
                }), 500
        else:
            return jsonify({
                "answer": f"Не найдено полигонов в радиусе {radius_km} км"
            }), 404
            
    except ValueError:
        return jsonify({"error": "Invalid coordinates format"}), 400
    except Exception as e:
        logger.error(f"Error in get_radius_intersection: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/species/synonyms", methods=["GET"])
def get_species_synonyms():
    """
    Возвращает все синонимы для заданного названия вида
    Пример: /species/synonyms?name=эдельвейс
    Пример: /species/synonyms?name=Leontopodium
    """
    species_name = request.args.get("name")
    if not species_name:
        return jsonify({"error": "Параметр 'name' обязателен"}), 400
    
    try:
        synonyms = search_service.get_synonyms_for_name(species_name)
        return jsonify(synonyms)
    except Exception as e:
        logger.error(f"Ошибка получения синонимов для '{species_name}': {str(e)}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500

@app.route("/species/all_synonyms", methods=["GET"])
def get_all_species_synonyms():
    """Возвращает все синонимы всех видов"""
    try:
        return jsonify(search_service.species_synonyms)
    except Exception as e:
        logger.error(f"Ошибка получения всех синонимов: {str(e)}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500
    
@app.route("/search_images_by_features", methods=["POST"])
def search_images_by_features():
    """
    Поиск изображений по признакам из feature_data
    Можно искать как по виду, так и только по признакам
    """
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    debug_info = {}
    logger.debug(debug_mode)
    logger.debug("\n")
    try:
        data = request.get_json()
        species_name = data.get("species_name")
        features = data.get("features", {})
        if "fruits_present" not in features:
            features["fruits_present"] = "нет"
            
        if not species_name and not features:
            return jsonify({"error": "Необходимо указать species_name или features"}), 400
        
        # Debug информация о запросе
        debug_info["request"] = {
            "species_name": species_name,
            "features": features,
            "timestamp": time.time()
        }
        
        if species_name:
            result = search_service.search_images_by_features(
                species_name=species_name,
                features=features
            )
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["search_type"] = "with_species"
                debug_info["synonyms_used"] = result.get("synonyms_used", {})
                debug_info["database_query"] = {
                    "species_conditions": result.get("species_conditions", []),
                    "feature_conditions": list(features.keys())
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                return jsonify(result), 404
            return jsonify(result)
        
        else:
            result = relational_service.search_images_by_features_only(
                features=features
            )
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["search_type"] = "features_only"
                debug_info["database_query"] = {
                    "feature_conditions": list(features.keys())
                }
                result["debug"] = debug_info
                
            if result.get("status") == "not_found":
                return jsonify(result), 404
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Ошибка поиска изображений по признакам: {str(e)}")
        error_response = {
            "status": "error",
            "message": f"Ошибка при поиске изображений: {str(e)}"
        }
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500

@app.route("/save_location", methods=["POST"])
def save_location():
    data = request.get_json()
    user_id = unquote(data.get("user_id"))
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if not user_id or not latitude or not longitude:
        return jsonify({"error": "Missing parameters"}), 400
    place_name = geo.reverse_geocode(float(latitude), float(longitude))
    user_locations[user_id] = {"latitude": latitude, "longitude": longitude, "place_name": place_name}
    return jsonify({"status": "ok"}), 200

@app.route("/objects_in_area_by_type", methods=["POST"])
def objects_in_area_by_type():
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    debug_info = {
        "timestamp": time.time(),
        "steps": []
    }
    
    data = request.get_json()
    area_name = data.get("area_name")
    object_type = data.get("object_type")
    object_subtype = data.get("object_subtype")
    object_name = data.get("object_name")
    limit = data.get("limit", 20)
    
    # Debug информация о параметрах
    debug_info["parameters"] = {
        "area_name": area_name,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "object_name": object_name,
        "limit": limit
    }
    
    # НОВАЯ ЛОГИКА: Если area_name не указан, но есть object_name - ищем объект напрямую
    if not area_name and object_name:
        debug_info["steps"].append({
            "step": "direct_object_search",
            "reason": "area_name not provided, searching object directly"
        })
        
        try:
            # Ищем объект по имени без привязки к области
            results = search_service.search_objects_directly_by_name(
                object_name=object_name,
                object_type=object_type,
                object_subtype=object_subtype,
                limit=limit
            )
            
            objects = results.get("objects", [])
            answer = results.get("answer", "")
            
            # Debug информация о результатах
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
            
            # Подготавливаем объекты для карты
            objects_for_map = []
            
            for obj in objects:
                name = obj.get('name', 'Без имени')
                description = obj.get('description', '')
                geojson = obj.get('geojson', {})
                obj_type = obj.get('type', 'unknown')
                
                popup_html = f"<h6>{name}</h6>"
                if obj_type:
                    popup_html += f"<p><strong>Тип:</strong> {obj_type}</p>"
                if description:
                    short_desc = description[:200] + "..." if len(description) > 200 else description
                    popup_html += f"<p>{short_desc}</p>"
                
                objects_for_map.append({
                    'tooltip': name,
                    'popup': popup_html,
                    'geojson': geojson
                })
            
            # Создаем карту
            map_name = f"direct_search_{object_name.replace(' ', '_')}_{int(time.time())}"
            map_result = geo.draw_custom_geometries(objects_for_map, map_name)
            
            map_result["count"] = len(objects)
            map_result["answer"] = answer
            map_result["objects"] = [{
                "name": obj.get('name'), 
                "description": obj.get('description'),
                "type": obj.get('type')
            } for obj in objects]
            
            # Добавляем debug информацию
            if debug_mode:
                debug_info["visualization"] = {
                    "map_name": map_name,
                    "total_objects_on_map": len(objects_for_map),
                    "search_type": "direct_object_search"
                }
                map_result["debug"] = debug_info
            
            return jsonify(map_result)
            
        except Exception as e:
            logger.error(f"Ошибка прямого поиска объекта: {str(e)}")
            debug_info["error"] = str(e)
            response = {"error": "Внутренняя ошибка сервера при поиске объекта"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
    
    # СТАРАЯ ЛОГИКА: Поиск по области (если area_name указан)
    if not area_name:
        response = {"error": "area_name is required when no object_name provided"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 400
    
    # Debug информация о параметрах
    debug_info["parameters"] = {
        "area_name": area_name,
        "object_type": object_type,
        "object_subtype": object_subtype,
        "object_name": object_name,
        "limit": limit
    }
    
    # Используем relational_service для поиска полигона области
    area_geometry = None
    area_info = None
    
    try:
        # Ищем полигон области через relational_service
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
        # Используем search_service для поиска объектов
        results = search_service.get_objects_in_area_by_type(
            area_geometry=area_geometry,
            object_type=object_type,
            object_subtype=object_subtype,
            object_name=object_name,
            limit=int(limit)
        )
        
        objects = results.get("objects", [])
        answer = results.get("answer", "")
        
        # Debug информация о результатах
        debug_info["search_results"] = {
            "total_objects_found": len(objects),
            "search_criteria": {
                "object_type": object_type,
                "object_subtype": object_subtype,
                "object_name": object_name
            }
        }
        
        if not objects:
            response = {
                "status": "no_objects", 
                "message": answer
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)
        
        # Подготавливаем объекты для карты
        objects_for_map = []
        
        # Добавляем полигон области как первый объект
        area_title = area_info.get('title', area_name) if area_info else area_name
        objects_for_map.append({
            'tooltip': f"Область поиска: {area_title}",
            'popup': f"<h6>{area_title}</h6><p>Область поиска</p>",
            'geojson': area_geometry
        })
        
        # Добавляем найденные объекты
        for obj in objects:
            name = obj.get('name', 'Без имени')
            description = obj.get('description', '')
            geojson = obj.get('geojson', {})
            obj_type = obj.get('type', 'unknown')
            
            popup_html = f"<h6>{name}</h6>"
            if obj_type:
                popup_html += f"<p><strong>Тип:</strong> {obj_type}</p>"
            if description:
                short_desc = description[:200] + "..." if len(description) > 200 else description
                popup_html += f"<p>{short_desc}</p>"
            
            objects_for_map.append({
                'tooltip': name,
                'popup': popup_html,
                'geojson': geojson
            })
        
        # Создаем карту
        map_name = f"area_search_{area_name.replace(' ', '_')}_{int(time.time())}"
        map_result = geo.draw_custom_geometries(objects_for_map, map_name)
        
        map_result["count"] = len(objects)
        map_result["answer"] = answer
        map_result["objects"] = [{
            "name": obj.get('name'), 
            "description": obj.get('description'),
            "type": obj.get('type')
        } for obj in objects]
        
        # Добавляем debug информацию
        if debug_mode:
            debug_info["visualization"] = {
                "map_name": map_name,
                "total_objects_on_map": len(objects_for_map),
                "area_included": True
            }
            map_result["debug"] = debug_info
        
        return jsonify(map_result)
        
    except Exception as e:
        logger.error(f"Ошибка поиска объектов по типу в области: {str(e)}")
        debug_info["error"] = str(e)
        response = {"error": "Внутренняя ошибка сервера при поиске"}
        if debug_mode:
            response["debug"] = debug_info
        return jsonify(response), 500
    
@app.route("/object/description/", methods=["GET", "POST"])
def get_object_description():
    # Обработка GET параметров
    object_name = request.args.get("object_name")
    query = request.args.get("query")
    limit = int(request.args.get("limit", 0))  # 0 = без лимита
    similarity_threshold = float(request.args.get("similarity_threshold", 0.01))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    use_gigachat_answer = request.args.get("use_gigachat_answer", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"
    object_type = request.args.get("object_type", "all")

    # Обработка POST body
    filter_data = None
    if request.method == "POST" and request.is_json:
        filter_data = request.get_json()
        logger.debug(f"Получены фильтры из body: {filter_data}")

    # ВАЖНО: Если use_gigachat_answer=True, то query обязателен
    if use_gigachat_answer and not query:
        return jsonify({"error": "Параметр 'query' обязателен при use_gigachat_answer=true"}), 400

    # Если нет ни object_name, ни query, ни filter_data - возвращаем ошибку
    if not object_name and not query and not filter_data:
        return jsonify({"error": "Необходимо указать object_name, query или передать фильтры в body"}), 400

    # Debug информация
    debug_info = {
        "parameters": {
            "object_name": object_name,
            "object_type": object_type,
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter,
            "use_gigachat_answer": use_gigachat_answer,
            "filter_data": filter_data
        },
        "timestamp": time.time()
    }

    try:
        # Определяем лимиты для разных случаев
        search_limit = limit if limit > 0 else 100  # Большой лимит для поиска
        context_limit = 5  # Топ описаний для контекста GigaChat
        
        if filter_data:
            # Используем новый метод для поиска по фильтрам
            descriptions = search_service.get_object_descriptions_by_filters(
                filter_data=filter_data,
                object_type=object_type,
                limit=search_limit
            )
            search_method = "filter_search"
            
        elif query:
            embedding = search_service.embedding_model.embed_query(query)
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding должен быть списком, получен: {type(embedding)}")
                return jsonify({"error": "Internal embedding error"}), 500
                
            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.error("Embedding содержит нечисловые элементы")
                return jsonify({"error": "Internal embedding error"}), 500
                
            # Если есть object_name - ищем по конкретному объекту
            if object_name:
                descriptions = search_service.get_object_descriptions_with_embedding(
                    object_name=object_name,
                    object_type=object_type,
                    query_embedding=embedding,
                    limit=search_limit,
                    similarity_threshold=similarity_threshold
                )
                search_method = "object_with_embedding"
            else:
                # Если нет object_name - ищем по семантическому сходству с query
                descriptions = search_service.search_objects_by_embedding(
                    query_embedding=embedding,
                    object_type=object_type,
                    limit=search_limit,
                    similarity_threshold=similarity_threshold
                )
                search_method = "semantic_search"
                
        else:
            # Если нет query, используем обычный поиск по object_name
            descriptions_text = search_service.get_object_descriptions(object_name, object_type)
            
            if include_similarity:
                descriptions = [{"content": text, "similarity": None, "source": "content"} 
                              for text in descriptions_text]
            else:
                descriptions = [{"content": text, "source": "content"} 
                              for text in descriptions_text]
            search_method = "simple_search"

        # Debug информация о результатах поиска
        if debug_mode:
            debug_info["search_method"] = search_method
            debug_info["search_results"] = {
                "total_found": len(descriptions),
                "search_limit": search_limit,
                "similarities": [desc.get("similarity", 0) for desc in descriptions] if descriptions and search_method != "simple_search" else []
            }

        # Обработка use_gigachat_filter
        if use_gigachat_filter:
            filter_query = query if query else object_name
            
            # Debug информация до фильтрации
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            
            filtered_descriptions = search_service.filter_text_descriptions_with_gigachat(
                filter_query, 
                descriptions
            )
            
            # Debug информация после фильтрации
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(filtered_descriptions),
                    "filtered_out": len(descriptions) - len(filtered_descriptions)
                }

            descriptions = filtered_descriptions

        # НОВАЯ ЛОГИКА: Генерация ответа GigaChat с обработкой blacklist
        if use_gigachat_answer:
            if not descriptions:
                response = {"error": "Не найдено описаний для генерации ответа"}
                if debug_mode:
                    response["debug"] = debug_info
                return jsonify(response), 404

            # Берем топ описаний для контекста (сортировка по similarity если есть)
            if all('similarity' in desc for desc in descriptions):
                context_descriptions = sorted(descriptions, key=lambda x: x.get('similarity', 0), reverse=True)[:context_limit]
            else:
                context_descriptions = descriptions[:context_limit]
            logger.info("=== ДОКУМЕНТЫ ДЛЯ GIGACHAT КОНТЕКСТА ===")
            for i, desc in enumerate(context_descriptions):
                content = desc["content"] if isinstance(desc, dict) else desc
                # Извлекаем первую строку как название
                first_line = content.strip().split('\n')[0] if content else "Без названия"
                title = first_line[:100]  # Ограничиваем длину для лога
                logger.info(f"Документ {i+1}: {title}")
                
                # Дополнительно логируем длину контента и первые 200 символов
                content_length = len(content) if content else 0
                preview = content[:200] + "..." if content and len(content) > 200 else content
                logger.info(f"  Длина: {content_length} символов")
                logger.info(f"  Превью: {preview}")
            logger.info("=== КОНЕЦ СПИСКА ДОКУМЕНТОВ ===")
            # Объединяем топ описаний в контекст
            context = "\n\n".join([
                desc["content"] if isinstance(desc, dict) else desc 
                for desc in context_descriptions
            ])
            
            # Добавляем информацию о количестве найденных записей
            total_count = len(descriptions)
            count_info = f"\n\nВсего найдено записей: {total_count}"
            if total_count > context_limit:
                count_info += f" (в контекст включено топ-{context_limit} по релевантности)"
            
            context += count_info
            logger.info(f"  Весь контекст: {context}")
            # Генерируем ответ с помощью GigaChat
            try:
                gigachat_result = search_service._generate_gigachat_answer(query, context)
                
                # Проверяем, был ли ответ заблокирован
                is_blacklist = gigachat_result.get("finish_reason") == "blacklist" or not gigachat_result.get("success", True)
                
                # Если ответ заблокирован, возвращаем форматированные описания вместо заглушки
                if is_blacklist:
                    logger.info("GigaChat вернул blacklist, возвращаем форматированные описания")
                    
                    # Форматируем описания так же, как в обычном ответе
                    formatted_descriptions = []
                    for i, desc in enumerate(descriptions, 1):
                        if isinstance(desc, dict):
                            content = desc.get("content", "")
                            similarity = desc.get("similarity")
                            source = desc.get("source", "unknown")
                            
                            formatted_desc = {
                                "id": i,
                                "content": content,
                                "source": source
                            }
                            
                            if similarity is not None:
                                formatted_desc["similarity"] = round(similarity, 4)
                                
                            lines = content.strip().split('\n')
                            if lines and lines[0].strip():
                                formatted_desc["title"] = lines[0].strip()[:100]
                            else:
                                formatted_desc["title"] = f"Описание {i}"
                                
                            formatted_descriptions.append(formatted_desc)
                        else:
                            formatted_descriptions.append({
                                "id": i,
                                "title": f"Описание {i}",
                                "content": desc,
                                "source": "content"
                            })

                    # Сортируем по similarity если есть
                    if all('similarity' in desc for desc in formatted_descriptions):
                        formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

                    response_data = {
                        "count": len(formatted_descriptions),
                        "descriptions": formatted_descriptions,
                        "query_used": query if query else "simple_search",
                        "similarity_threshold": similarity_threshold if query else None,
                        "use_gigachat_filter": use_gigachat_filter,
                        "use_gigachat_answer": True,
                        "gigachat_restricted": True,
                        "message": "GigaChat не смог сгенерировать ответ, поэтому возвращены исходные описания",
                        "formatted": True
                    }

                    if object_name:
                        response_data["object_name"] = object_name
                        response_data["object_type"] = object_type

                    if filter_data:
                        response_data["filters_applied"] = filter_data

                    if debug_mode:
                        response_data["debug"] = debug_info
                        response_data["debug"]["gigachat_generation"] = {
                            "finish_reason": gigachat_result.get("finish_reason"),
                            "blacklist_detected": True,
                            "fallback_to_descriptions": True
                        }

                    return jsonify(response_data)
                
                # Если ответ не заблокирован, возвращаем обычный ответ GigaChat
                gigachat_response = gigachat_result.get("content", "")
                
                response_data = {
                    "gigachat_answer": gigachat_response,
                    "context_used": {
                        "descriptions_count": len(context_descriptions),
                        "total_descriptions": total_count,
                        "total_characters": len(context)
                    },
                    "query": query,
                    "object_name": object_name if object_name else "semantic_search",
                    "object_type": object_type
                }
                
                # Добавляем debug информацию
                if debug_mode:
                    response_data["debug"] = debug_info
                    response_data["debug"]["gigachat_generation"] = {
                        "context_samples": [desc["content"][:200] + "..." if len(desc["content"]) > 200 else desc["content"] 
                                          for desc in context_descriptions[:2]],
                        "response_length": len(gigachat_response),
                        "finish_reason": gigachat_result.get("finish_reason"),
                        "blacklist_detected": False
                    }

                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Ошибка генерации ответа GigaChat: {str(e)}")
                error_response = {"error": "Ошибка генерации ответа GigaChat"}
                if debug_mode:
                    debug_info["gigachat_error"] = str(e)
                    error_response["debug"] = debug_info
                return jsonify(error_response), 500

        # УЛУЧШЕННАЯ ЛОГИКА: Форматированный ответ без GigaChat
        if not descriptions:
            response = {"error": "Описание не найдено"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        # Форматируем описания в красивый структурированный вид
        formatted_descriptions = []
        for i, desc in enumerate(descriptions, 1):
            if isinstance(desc, dict):
                content = desc.get("content", "")
                similarity = desc.get("similarity")
                source = desc.get("source", "unknown")
                
                # Создаем структурированную запись
                formatted_desc = {
                    "id": i,
                    "content": content,
                    "source": source
                }
                
                if similarity is not None:
                    formatted_desc["similarity"] = round(similarity, 4)
                    
                # Извлекаем название из контента (первая строка или первые 50 символов)
                lines = content.strip().split('\n')
                if lines and lines[0].strip():
                    formatted_desc["title"] = lines[0].strip()[:100]
                else:
                    formatted_desc["title"] = f"Описание {i}"
                    
                formatted_descriptions.append(formatted_desc)
            else:
                # Для простых строк
                formatted_descriptions.append({
                    "id": i,
                    "title": f"Описание {i}",
                    "content": desc,
                    "source": "content"
                })

        # Сортируем по similarity если есть
        if all('similarity' in desc for desc in formatted_descriptions):
            formatted_descriptions.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        response_data = {
            "count": len(formatted_descriptions),
            "descriptions": formatted_descriptions,
            "query_used": query if query else "simple_search",
            "similarity_threshold": similarity_threshold if query else None,
            "use_gigachat_filter": use_gigachat_filter,
            "formatted": True  # Флаг что ответ отформатирован
        }

        # Добавляем информацию об объекте, если он был указан
        if object_name:
            response_data["object_name"] = object_name
            response_data["object_type"] = object_type

        # Добавляем информацию о фильтрах, если они были использованы
        if filter_data:
            response_data["filters_applied"] = filter_data

        # Добавляем debug информацию
        if debug_mode:
            response_data["debug"] = debug_info

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Ошибка получения описания: {str(e)}", exc_info=True)
        error_response = {"error": "Внутренняя ошибка сервера"}
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
    
@app.route("/species/description/", methods=["GET"])
def get_species_description():
    species_name = request.args.get("species_name")
    query = request.args.get("query")
    limit = int(request.args.get("limit", 5))
    similarity_threshold = float(request.args.get("similarity_threshold", 0.5))
    include_similarity = request.args.get("include_similarity", "false").lower() == "true"
    use_gigachat_filter = request.args.get("use_gigachat_filter", "false").lower() == "true"
    debug_mode = request.args.get("debug_mode", "false").lower() == "true"

    if not species_name:
        return jsonify({"error": "species_name parameter is required"}), 400

    # Debug информация
    debug_info = {
        "parameters": {
            "species_name": species_name,
            "query": query,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            "include_similarity": include_similarity,
            "use_gigachat_filter": use_gigachat_filter
        },
        "timestamp": time.time()
    }

    try:
        if query:
            embedding = search_service.embedding_model.embed_query(query)
            
            if not isinstance(embedding, list):
                logger.error(f"Embedding должен быть списком, получен: {type(embedding)}")
                return jsonify({"error": "Internal embedding error"}), 500
                
            if not all(isinstance(x, (int, float)) for x in embedding):
                logger.error("Embedding содержит нечисловые элементы")
                return jsonify({"error": "Internal embedding error"}), 500
                
            # Debug информация об embedding
            if debug_mode:
                debug_info["embedding"] = {
                    "type": type(embedding).__name__,
                    "length": len(embedding),
                    "first_5_elements": embedding[:5] if isinstance(embedding, list) else "N/A"
                }
            
            descriptions = relational_service.get_text_descriptions_with_embedding(
                species_name=species_name,
                query_embedding=embedding,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # Debug информация о результатах поиска
            if debug_mode:
                debug_info["search_method"] = "embedding_similarity"
                debug_info["search_results"] = {
                    "total_found": len(descriptions),
                    "similarities": [desc.get("similarity", 0) for desc in descriptions] if descriptions else []
                }
            
            if not include_similarity:
                descriptions = [{"content": desc["content"], "source": desc.get("source", "unknown")} 
                              for desc in descriptions]
            else:
                descriptions = [{"content": desc["content"], 
                               "similarity": desc["similarity"],
                               "source": desc.get("source", "unknown")} 
                              for desc in descriptions]
                
        else:
            descriptions_text = search_service.get_text_descriptions(species_name)
            
            # Debug информация
            if debug_mode:
                debug_info["search_method"] = "simple_search"
                debug_info["search_results"] = {
                    "total_found": len(descriptions_text)
                }
            
            if include_similarity:
                descriptions = [{"content": text, "similarity": None, "source": "content"} 
                              for text in descriptions_text]
            else:
                descriptions = [{"content": text, "source": "content"} 
                              for text in descriptions_text]
                
        if use_gigachat_filter:
            filter_query = query if query else species_name
            
            logger.debug("Описания для фильтрации с Gigachat")
            logger.debug(descriptions)
            
            # Debug информация до фильтрации
            if debug_mode:
                debug_info["before_gigachat_filter"] = {
                    "count": len(descriptions),
                    "filter_query": filter_query
                }
            
            filtered_descriptions = search_service.filter_text_descriptions_with_gigachat(
                filter_query, 
                descriptions
            )
            
            # Debug информация после фильтрации
            if debug_mode:
                debug_info["after_gigachat_filter"] = {
                    "count": len(filtered_descriptions),
                    "filtered_out": len(descriptions) - len(filtered_descriptions)
                }

            if not include_similarity:
                descriptions = [desc["content"] for desc in filtered_descriptions]
            else:
                descriptions = filtered_descriptions

        if not descriptions:
            response = {"error": "Описание не найдено"}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 404

        # Форматируем ответ в зависимости от параметров
        if include_similarity:
            response_data = {
                "count": len(descriptions),
                "descriptions": descriptions,
                "query_used": query if query else "simple_search",
                "similarity_threshold": similarity_threshold if query else None,
                "use_gigachat_filter": use_gigachat_filter
            }
        else:
            response_data = {
                "count": len(descriptions),
                "descriptions": [desc["content"] if isinstance(desc, dict) else desc for desc in descriptions],
                "query_used": query if query else "simple_search",
                "similarity_threshold": similarity_threshold if query else None,
                "use_gigachat_filter": use_gigachat_filter
            }

        # Добавляем debug информацию
        if debug_mode:
            response_data["debug"] = debug_info

        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Ошибка получения описания для '{species_name}': {str(e)}", exc_info=True)
        error_response = {"error": "Внутренняя ошибка сервера"}
        if debug_mode:
            debug_info["error"] = str(e)
            error_response["debug"] = debug_info
        return jsonify(error_response), 500
    

@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        logger.debug(data)
        question = data.get("question", "")
        use_agent = data.get("use_agent", False)
        knowledge_base_type = data.get("knowledge_base_type", "vector")
        strict_filter = data.get("strict_filter", False)
        filter_words = data.get("filter_words")
        
        if use_agent and knowledge_base_type != "relational":
            return jsonify({
                "error": "Для использования агента необходимо knowledge_base_type='relational'"
            }), 400

        result = search_service.ask_question(
            question=question,
            similarity_threshold=float(data.get("similarity_threshold", 0.5)),
            similarity_deviation=float(data.get("similarity_deviation")) if "similarity_deviation" in data else None,
            use_gigachat=data.get("use_gigachat", False),
            user_id=data.get("user_id"),
            debug_mode=data.get("debug_mode", False),
            knowledge_base_type=knowledge_base_type,
            query_formatter="agent" if use_agent else data.get("query_formatter"),
            strict_filter=strict_filter,
            filter_words=filter_words  
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_location", methods=["GET"])
def get_location():
    user_id = unquote(request.args.get("user_id"))
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    location = user_locations.get(user_id)
    if not location:
        return jsonify({"error": "Location not found"}), 404
    return jsonify(location)

@app.route("/is_known_object", methods=["POST"])
def handle_check():
    data = request.json
    object_name = data.get("object", "")
    result = slot_val.is_known_object(object_name)
    return jsonify(result)

@app.route("/get_coords", methods=["POST"])
def api_get_coords():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"status": "error", "message": "Параметр 'name' обязателен."}), 400

    result = geo.get_point_coords_from_geodb(name)
    return jsonify(result)

@app.route("/visualize_objects", methods=["POST"])
def api_visualize_objects():
    data = request.get_json()
    objects = data.get("objects")
    if not objects:
        return jsonify({"status": "error", "message": "Нет объектов для визуализации"}), 400

    result = geo.draw_custom_geometries(objects)
    return jsonify(result)

@app.route("/coords_to_map", methods=["POST"])
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
    
    logger.debug(f"""Параметры:{data}""")
    if not lat or not lon:
        return jsonify({"status": "error", "message": "Не заданы координаты."}), 400

    # Initialize t3 in case visualization fails
    t3 = time.perf_counter()
    
    try:
        t1 = time.perf_counter()
        result = search_service.get_nearby_objects(
            latitude=float(lat),
            longitude=float(lon),
            radius_km=float(radius),
            object_type=object_type,
            species_name=species_name  
        )
        t2 = time.perf_counter()
        objects = result.get("objects", [])
        answer = result.get("answer", "")
        logger.debug(f"""Найденные объекты: {objects}""")
        
        # Debug информация
        debug_info = {
            "search_time": round(t2 - t1, 3),
            "parse_time": round(t_after_parse - t0, 3),
            "parameters": {
                "latitude": lat,
                "longitude": lon,
                "radius_km": radius,
                "object_type": object_type,
                "species_name": species_name
            },
            "objects_count": len(objects),
            "search_query_details": result.get("debug_info", {})
        }
        
        if not objects:
            response = {"status": "no_objects", "message": answer}
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)

        # Filter out invalid geometries before visualization
        valid_objects = []
        object_details = []
        for obj in objects:
            try:
                if obj.get("geojson") and obj["geojson"].get("coordinates"):
                    # Basic validation of coordinates
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
                    elif isinstance(obj["geojson"]["coordinates"][0], list):
                        # For polygons/multipoints, check first coordinate
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
            except Exception as e:
                logger.warning(f"Invalid geometry in object {obj.get('name')}: {str(e)}")
                continue

        debug_info["valid_objects_count"] = len(valid_objects)
        debug_info["object_details"] = object_details
        debug_info["validation_errors"] = len(objects) - len(valid_objects)

        if not valid_objects:
            response = {
                "status": "error",
                "message": "Найдены объекты, но их координаты недействительны для отображения"
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response)

        # 2. Визуализируем только валидные объекты
        try:
            map_result = geo.draw_custom_geometries(valid_objects, "custom_map")
            t3 = time.perf_counter()
            map_result["count"] = len(valid_objects)
            map_result["answer"] = answer
            map_result["names"] = [obj.get("name", "Без имени") for obj in valid_objects]
            
            # Добавляем debug информацию
            debug_info["render_time"] = round(t3 - t2, 3)
            debug_info["total_time"] = round(time.perf_counter() - t0, 3)
            debug_info["map_generation"] = {
                "static_map": map_result.get("static_map"),
                "interactive_map": map_result.get("interactive_map")
            }
            
            if debug_mode:
                map_result["debug"] = debug_info
                
            return jsonify(map_result)
        except Exception as e:
            logger.error(f"Ошибка отрисовки карты: {e}")
            debug_info["render_error"] = str(e)
            response = {
                "status": "error", 
                "message": f"Ошибка отрисовки карты: {e}",
                "objects": [obj["name"] for obj in valid_objects],
                "answer": answer
            }
            if debug_mode:
                response["debug"] = debug_info
            return jsonify(response), 500
            
    except Exception as e:
        logger.error(f"Ошибка поиска рядом: {e}")
        debug_info["search_error"] = str(e)
        response = {"status": "error", "message": f"Ошибка поиска рядом: {e}"}
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

@app.route("/draw_multiple_places", methods=["POST"])
def draw_multiple_places():
    data = request.get_json()
    places = data.get("geo_places")

    if not places or not isinstance(places, list):
        return jsonify({"status": "error", "message": "Параметр 'geo_places' должен быть списком"}), 400

    try:
        objects = []
        for p in places:
            match = find_place_flexible(p)
            if match:
                objects.append({
                    "name": p,
                    "geojson": match["record"]["geometry"]
                })
        if not objects:
            return jsonify({"status": "error", "message": "Не удалось найти геометрию ни для одного объекта."}), 404

        result = geo.draw_custom_geometries(objects, "multi_places_map")

        if result["status"] != "ok":
            return jsonify({"status": "error", "message": "Не удалось отрисовать карту"}), 500

        return jsonify({
            "status": "ok",
            "static_map": result["static_map"],
            "interactive_map": result["interactive_map"],
            "count": len(objects),
            "answer": f"На карте отображены {len(objects)} геообъектов."
        })

    except Exception as e:
        return jsonify({"status": "error", "message": f"Ошибка: {e}"}), 500

@app.route('/to_prepositional_case', methods=['POST'])
def to_prepositional_case():
    data = request.get_json()
    place_raw = data.get("place", "")
    object_raw = data.get("object", "")

    place_data = find_place_flexible(place_raw)["name"]
    if place_data == "not_found":
        return jsonify({
            "status": "not_found",
            "answer": "not found geo place"
        }), 404

    object_prep = to_prepositional_phrase(object_raw)

    return jsonify({
        "status": "ok",
        "place_normalized": place_data,
        "object_prepositional": object_prep,
    }), 200

@app.route("/find_species_with_description", methods=["POST"])
def find_species_with_description():
    data = request.get_json()
    name = data.get("name")
    limit = data.get("limit", 5)
    offset = data.get("offset", 0)

    result = slot_val.find_species_with_description(name, limit, offset) 
    return jsonify(result)

@app.route("/")
def home():
    return "SalutBot API works!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)