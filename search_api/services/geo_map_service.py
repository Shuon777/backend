import os
from typing import Dict, Any, List, Tuple

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import shape, GeometryCollection, mapping

from ..domain.value_objects import GeoContent, MapLinks


class GeoMapService:
    def __init__(self, maps_dir: str, domain: str):
        self.maps_dir = maps_dir
        self.domain = domain
        os.makedirs(maps_dir, exist_ok=True)

    def _add_basemap(self, ax) -> None:
        sources = [
            ctx.providers.Esri.WorldImagery,
            ctx.providers.CartoDB.Positron,
        ]
        for source in sources:
            try:
                ctx.add_basemap(ax, source=source)
                return
            except Exception:
                continue

    def generate_static_map(self, geojson: Dict[str, Any], name: str) -> str:
        geom = shape(geojson)
        if geom.is_empty:
            return ""

        if geom.geom_type == "GeometryCollection":
            geometries = list(geom.geoms)
        else:
            geometries = [geom]

        gdf = gpd.GeoDataFrame(
            [{"geometry": g} for g in geometries],
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

        for _, row in gdf.iterrows():
            g = row.geometry
            if g.geom_type == "Point":
                gdf_point = gpd.GeoDataFrame([row], crs=gdf.crs)
                gdf_point.plot(ax=ax, marker='o', markersize=30,
                              color='red', edgecolor='darkred',
                              linewidth=1, alpha=0.7, zorder=10)
            else:
                gdf_poly = gpd.GeoDataFrame([row], crs=gdf.crs)
                gdf_poly.plot(ax=ax, facecolor='white',
                             edgecolor='darkblue', linewidth=2,
                             alpha=0.5, zorder=5)

        if geometries:
            bounds = gdf.total_bounds
            buffer = max(3000, (bounds[2] - bounds[0]) * 0.1)
            ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
            ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)

        self._add_basemap(ax)
        ax.axis('off')

        filename = f"{name}.jpeg"
        filepath = os.path.join(self.maps_dir, filename)
        plt.savefig(filepath, format='jpeg', dpi=150,
                   bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        return f"{self.domain}/maps/{filename}"

    def generate_interactive_map(self, geojson: Dict[str, Any], name: str) -> str:
        geom = shape(geojson)
        centroid = geom.centroid
        m = folium.Map(location=[centroid.y, centroid.x],
                      zoom_start=9, tiles='OpenStreetMap',
                      attributionControl=False)
        folium.GeoJson(mapping(geom), tooltip=name, name=name).add_to(m)
        filename = f"webapp_{name}.html"
        filepath = os.path.join(self.maps_dir, filename)
        m.save(filepath)
        return f"{self.domain}/maps/{filename}"

    def enrich_geo_content(self, geojson: Dict[str, Any], name: str) -> GeoContent:
        static_url = self.generate_static_map(geojson, name)
        interactive_url = self.generate_interactive_map(geojson, name)
        geom = shape(geojson)
        return GeoContent(
            geojson=geojson,
            geometry_type=geom.geom_type,
            map_links=MapLinks(static=static_url, interactive=interactive_url)
        )