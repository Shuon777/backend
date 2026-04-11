from typing import List
from ..domain.entities import ResourceResult
from ..domain.value_objects import ModalityType, GeoContent
from .geo_map_service import GeoMapService


class ResponseEnricher:
    def __init__(self, maps_dir: str, domain: str):
        self._geo_service = GeoMapService(maps_dir, domain)

    def enrich_resources(self, resources: List[ResourceResult]) -> List[ResourceResult]:
        enriched = []
        for resource in resources:
            if resource.modality_type == ModalityType.GEODATA.value:
                enriched.append(self._enrich_geo_resource(resource))
            else:
                enriched.append(resource)
        return enriched

    def _enrich_geo_resource(self, resource: ResourceResult) -> ResourceResult:
        content = resource.content
        if not isinstance(content, dict):
            return resource

        geojson = content.get('geojson')
        if not geojson:
            return resource

        name = resource.title or f"map_{resource.id}"
        geo_content = self._geo_service.enrich_geo_content(geojson, name)

        return ResourceResult(
            id=resource.id,
            title=resource.title,
            uri=resource.uri,
            author=resource.author,
            source=resource.source,
            modality_type=resource.modality_type,
            content=geo_content,
            features=resource.features
        )