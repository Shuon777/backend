import os
import time
import traceback
from urllib.parse import quote
from typing import Tuple, List, Dict, Any

import requests
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import contextily as ctx
import folium
import pyproj
from shapely.geometry import shape, Point, GeometryCollection, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from infrastructure.geo_db_store import get_place, add_place
from infrastructure.maps_store import set_map_links

matplotlib.use('Agg')


class GeoProcessor:
    def __init__(self, maps_dir: str, domain: str):
        self.maps_dir = maps_dir
        self.domain = domain
        os.makedirs(self.maps_dir, exist_ok=True)

    def add_basemap(self, ax: matplotlib.axes.Axes) -> None:
        for source in [
            ctx.providers.Esri.WorldImagery,
            ctx.providers.CartoDB.Positron,
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}",
        ]:
            try:
                ctx.add_basemap(ax, source=source)
                return
            except Exception:
                continue

    def buffer_km(self, geom: BaseGeometry, buffer_km: float) -> BaseGeometry:
        proj_wgs84 = pyproj.CRS('EPSG:4326')
        proj_3857 = pyproj.CRS('EPSG:3857')
        to_3857 = pyproj.Transformer.from_crs(proj_wgs84, proj_3857, always_xy=True).transform
        to_4326 = pyproj.Transformer.from_crs(proj_3857, proj_wgs84, always_xy=True).transform
        geom_3857 = transform(to_3857, geom)
        buffer_geom_3857 = geom_3857.buffer(buffer_km * 1000)
        return transform(to_4326, buffer_geom_3857)

    def generate_folium_map(self, geometry: BaseGeometry, place_name: str) -> str:
        if geometry.geom_type == "Point":
            lat, lon = geometry.y, geometry.x
            m = folium.Map(location=[lat, lon], zoom_start=10, tiles='OpenStreetMap')
            folium.Marker([lat, lon], popup=place_name).add_to(m)
        else:
            bounds = geometry.bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles='OpenStreetMap')
            folium.GeoJson(
                geometry.__geo_interface__,
                name=place_name,
                tooltip=place_name  
            ).add_to(m)

        filename_html = f"webapp_{place_name}.html"
        filepath_html = os.path.join(self.maps_dir, filename_html)
        m.save(filepath_html)
        return f"{self.domain}/maps/{filename_html}"

    def draw_geometry(self, geometry: BaseGeometry, place_name: str) -> Tuple[str, str]:
        # Распаковываем GeometryCollection
        if isinstance(geometry, GeometryCollection):
            geometries = list(geometry.geoms)
        else:
            geometries = [geometry]

        # Строим GeoDataFrame из всех геометрий
        gdf = gpd.GeoDataFrame([{"geometry": g} for g in geometries], crs="EPSG:4326").to_crs(epsg=3857)

        # Отрисовка
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, facecolor='skyblue', edgecolor='black', alpha=0.5)

        # Масштабирование с отступами
        minx, miny, maxx, maxy = gdf.total_bounds
        buffer = 5000
        ax.set_xlim(minx - buffer, maxx + buffer)
        ax.set_ylim(miny - buffer, maxy + buffer)

        self.add_basemap(ax)
        ax.axis('off')
        plt.tight_layout(pad=0)

        # Сохранение карты
        filename = f"{place_name}.png"
        image_path = os.path.join(self.maps_dir, filename)
        plt.savefig(image_path, dpi=250, bbox_inches='tight', pad_inches=0)
        plt.close()

        static_map_url = f"{self.domain}/maps/{filename}"
        web_app_url = self.generate_folium_map(geometry, place_name)

        set_map_links(place_name, {
            "static": static_map_url,
            "interactive": web_app_url
        })

        return static_map_url, web_app_url

    def fetch_and_draw(self, place: str, flag_if_exist: bool) -> List[Dict[str, Any]]:
       
        existing = get_place(place)
        if existing:
            geometry = shape(existing["geometry"])
            if flag_if_exist:
                self.draw_geometry(geometry, place)  
            return [{"geometry": existing["geometry"]}]

        
        print(f"🔍 Ищем в OSM: {place}")
        encoded_place = quote(place)
        url = f"https://nominatim.openstreetmap.org/search?q={encoded_place}&format=json&polygon_geojson=1"
        headers = {"User-Agent": "BaikalGeo/1.0"}

        try:
            response = requests.get(url, headers=headers)
            time.sleep(1.2)
            features = []
            if response.status_code == 200:
                results = response.json()
                if results:
                    for result in results:
                        geometry = result.get("geojson") or {
                            "type": "Point",
                            "coordinates": [float(result["lon"]), float(result["lat"])]
                        }
                        geom = shape(geometry)
                        self.draw_geometry(geom, place)
                        record = {"name": place, "geometry": geometry}
                        add_place(place, record)
                        features.append({"geometry": geometry})
                    return features

            
            cached = get_place(place)
            if cached:
                geometry = shape(cached["geometry"])
                self.draw_geometry(geometry, place)
                return [{"geometry": cached["geometry"]}]

            return []

        except Exception:
            traceback.print_exc()
            return []



    def fetch_and_draw_multiple(self, places: List[str]) -> Dict[str, Any]:
        geoms = []
        for place in places:
            features = self.fetch_and_draw(place, True)
            if features:
                geoms.append(shape(features[0]["geometry"]))

        if not geoms:
            return {"status": "no_geometries", "answer": "Не удалось найти геометрии для выбранных мест."}

        intersection_geom = geoms[0]
        for g in geoms[1:]:
            intersection_geom = intersection_geom.intersection(g)

        if intersection_geom.is_empty:
            return {"status": "no_intersection", "answer": "Области не пересекаются. Карта показывает пустую область."}

        name = "_".join([p.replace(" ", "_") for p in places]) + "_intersection"
        static_map_url, web_app_url = self.draw_geometry(intersection_geom, name)
        return {
            "status": "ok",
            "map_image": static_map_url,
            "web_app_url": web_app_url,
            "answer": "Найдено пересечение областей для выбранных мест."
        }

    def get_species_area_near_center(self, center_name: str, region_name: str, buffer_km_val: float = 10) -> Dict[str, Any]:
        center_features = self.fetch_and_draw(center_name, False)
        if not center_features:
            return {"status": "no_center_found", "answer": f"Не удалось найти геометрию для {center_name}."}
        center_geom = shape(center_features[0]["geometry"])

        region_features = self.fetch_and_draw(region_name, False)
        if not region_features:
            return {"status": "no_region_found", "answer": f"Не удалось найти геометрию для {region_name}."}
        region_geom = shape(region_features[0]["geometry"])

        buffer_geom = self.buffer_km(center_geom, buffer_km_val)
        search_zone = buffer_geom.intersection(region_geom)

        if search_zone.is_empty:
            return {"status": "no_intersection", "answer": f"Область поиска вокруг {center_name} не пересекает {region_name}."}

        name = f"{center_name}_{region_name}_search_zone"
        static_map_url, web_app_url = self.draw_geometry(search_zone, name)
        return {
            "status": "ok",
            "map_image": static_map_url,
            "web_app_url": web_app_url,
            "answer": f"Найдена область поиска для {center_name} с радиусом {buffer_km_val} км в пределах {region_name}."
        }

    def reverse_geocode(self, lat: float, lon: float) -> str:
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1"
            headers = {"User-Agent": "TestEcoBot (testecobot.ru)"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return "Не удалось определить место"
            data = response.json()
            address = data.get("address", {})
            return next(
                (comp for comp in [address.get(k) for k in [
                    "city", "town", "village", "municipality", "county", "state", "region", "country"]] if comp),
                "Неизвестное место"
            )
        except Exception as e:
            print(f"Ошибка reverse_geocode: {e}")
            return "Не удалось определить место"
    
    def get_point_coords_from_geodb(self, name: str) -> dict:
        entry = get_place(name)
        if not entry:
            return {"status": "not_found", "message": f"Объект '{name}' не найден."}

        geom = shape(entry["geometry"])
        
        if isinstance(geom, Point):
            lat, lon = geom.y, geom.x
        else:
            # Для Polygon / LineString — взять центр
            centroid = geom.centroid
            lat, lon = centroid.y, centroid.x

        return {
            "status": "ok",
            "latitude": lat,
            "longitude": lon
        }
    
    # In coordinates_finder.py

    def draw_custom_geometries(self, objects: List[dict], name: str) -> dict:
        from shapely.geometry import shape, GeometryCollection, mapping

        if not objects:
            return {"status": "error", "message": "Нет объектов для отрисовки"}

        geometries = []
        tooltips = []
        popups = []

        for obj in objects:
            geojson = obj.get("geojson")
            if not geojson:
                continue
            try:
                # Геометрия
                geom = shape(geojson)
                geometries.append(geom)
                
                # Тексты для карты.
                # Используем 'tooltip' и 'popup', если они есть, иначе 'name'.
                tooltips.append(obj.get("tooltip", obj.get("name", "Без имени")))
                popups.append(obj.get("popup", obj.get("name", "Без имени")))

            except Exception as e:
                print(f"Ошибка в geojson или при его обработке: {e}")

        if not geometries:
            return {"status": "error", "message": "Нет валидных геометрий для отрисовки"}

        # --- Создание статической карты (без изменений) ---
        combined_static = GeometryCollection(geometries)
        static_map, _ = self.draw_geometry(combined_static, name)

        # --- Создание интерактивной карты Folium ---
        centroid = combined_static.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=9, tiles="OpenStreetMap")

        # Добавляем геометрии с кастомными tooltip и popup
        for geom, tooltip_text, popup_html in zip(geometries, tooltips, popups):
            # Folium.Popup позволяет рендерить HTML
            popup = folium.Popup(popup_html, max_width=400)

            folium.GeoJson(
                mapping(geom),
                tooltip=tooltip_text, # Текст при наведении
                popup=popup           # Окно с HTML при клике
            ).add_to(m)

        filename_html = f"webapp_{name}.html"
        filepath_html = os.path.join(self.maps_dir, filename_html)
        m.save(filepath_html)
        interactive_map_url = f"{self.domain}/maps/{filename_html}"

        return {
            "status": "ok",
            "static_map": static_map,
            "interactive_map": interactive_map_url
        }

    def draw_custom_geometries_two(self, geoms: List[dict], name: str) -> dict:
        from shapely.geometry import shape, GeometryCollection, mapping

        geometries = []
        names = []
        for geo in geoms:
            geojson = geo
            if not geojson:
                continue
            try:
                geom = shape(geojson)
                geometries.append(geom)
                names.append(geo.get("name", "Без имени"))
            except Exception as e:
                print(f"Ошибка в geojson: {e}")

        if not geometries:
            return {"status": "error", "message": "Нет валидных геометрий"}

        combined = GeometryCollection(geometries)
        static_map, _ = self.draw_geometry(combined, name)

        centroid = combined.centroid
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=9, tiles="OpenStreetMap")

        for geom, title in zip(geometries, names):
            folium.GeoJson(
                mapping(geom),
                tooltip=title
            ).add_to(m)

        filename_html = f"webapp_{name}.html"
        filepath_html = os.path.join(self.maps_dir, filename_html)
        m.save(filepath_html)
        interactive_map_url = f"{self.domain}/maps/{filename_html}"

        return {
            "status": "ok",
            "static_map": static_map,
            "interactive_map": interactive_map_url
        } 
    

