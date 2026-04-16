import logging
from typing import List, Optional
from sqlalchemy.orm import joinedload
from sqlalchemy import func,literal
from sqlalchemy.dialects import postgresql
from ..domain.entities import ObjectResult, ResourceResult, ObjectCriteria, ResourceCriteria
from .search_repository import SearchRepository
from ..infrastructure.orm.object_models import Object, ObjectNameSynonym, ObjectType
from ..infrastructure.orm.resource_models import Resource, Bibliographic, Author, Source, ResourceStatic
from ..infrastructure.orm.modality_models import Modality, TextValue, ImageValue, GeodataValue, ResourceValue


logger = logging.getLogger(__name__)

class SQLAlchemySearchRepository(SearchRepository):
    def __init__(self, session_factory):
        self._session_factory = session_factory

    def find_objects_by_criteria(self, criteria: ObjectCriteria, limit: int = 20, offset: int = 0) -> List[ObjectResult]:
        if not criteria.db_id and not criteria.name_synonyms and not criteria.properties and not criteria.object_type:
            return []
        
        session = self._session_factory()
        with session:
            query = session.query(Object).options(joinedload(Object.synonyms)).join(Object.object_type)
            
            if criteria.db_id:
                query = query.filter(Object.db_id == criteria.db_id)
            if criteria.object_type:
                query = query.filter(ObjectType.name == criteria.object_type)
            if criteria.name_synonyms:
                names = []
                for lang, name_list in criteria.name_synonyms.items():
                    names.extend(name_list)
                if names:
                    query = query.filter(Object.synonyms.any(ObjectNameSynonym.synonym.in_(names)))
            if criteria.properties:
                for key, value in criteria.properties.items():
                    if key == 'subtypes':
                        if isinstance(value, str):
                            query = query.filter(Object.object_properties[key].op('?')(value))
                        elif isinstance(value, list):
                            for item in value:
                                query = query.filter(Object.object_properties[key].op('?')(item))
                        else:
                            query = query.filter(Object.object_properties[key].as_string() == str(value))
                    else:
                        if isinstance(value, str):
                            query = query.filter(Object.object_properties[key].as_string() == value)
                        elif isinstance(value, list):
                            for item in value:
                                query = query.filter(Object.object_properties[key].as_string() == item)
                        elif isinstance(value, bool):
                            query = query.filter(Object.object_properties[key].as_boolean() == value)
                        elif isinstance(value, (int, float)):
                            query = query.filter(Object.object_properties[key].as_float() == value)
                        else:
                            query = query.filter(Object.object_properties[key].as_string() == str(value))
            
            query = query.limit(limit).offset(offset)
            
            compiled = query.statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
            logger.info(f"Executing query: {compiled}")
            
            objects = query.all()
            
            return [
                ObjectResult(
                    id=obj.id,
                    db_id=obj.db_id,
                    object_type=obj.object_type.name,
                    properties=obj.object_properties,
                    synonyms=[s.synonym for s in obj.synonyms]
                ) for obj in objects
            ]
            
    def find_resources_by_criteria(self, criteria: ResourceCriteria, object_ids: Optional[List[int]] = None, limit: int = 50, offset: int = 0) -> List[ResourceResult]:
        session = self._session_factory()
        with session:
            query = session.query(Resource).outerjoin(
                ResourceStatic, Resource.resource_static_id == ResourceStatic.id
            ).outerjoin(
                Bibliographic, ResourceStatic.bibliographic_id == Bibliographic.id
            ).outerjoin(
                Author, Bibliographic.author_id == Author.id
            ).outerjoin(
                Source, Bibliographic.source_id == Source.id
            )
            
            if object_ids:
                query = query.filter(Resource.objects.any(Object.id.in_(object_ids)))
            if criteria.title:
                query = query.filter(Resource.title.ilike(f"%{criteria.title}%"))
            if criteria.uri:
                query = query.filter(Resource.uri == criteria.uri)
            if criteria.author:
                query = query.filter(Author.name.ilike(f"%{criteria.author}%"))
            if criteria.source:
                query = query.filter(Source.name.ilike(f"%{criteria.source}%"))
            if criteria.modality_type:
                query = query.filter(Resource.resource_values.any(Modality.modality_type == criteria.modality_type))
            if criteria.features:
                for key, val in criteria.features.items():
                    query = query.filter(Resource.features[key].as_string() == str(val))
            
            resources = query.limit(limit).offset(offset).all()
            result = []
            for r in resources:
                rv = r.resource_values[0] if r.resource_values else None
                content = None
                if rv and rv.modality:
                    mt = rv.modality.modality_type
                    if mt == 'Текст' and rv.value_id:
                        tv = session.query(TextValue).get(rv.value_id)
                        content = {'structured_data': tv.structured_data} if tv else None
                    elif mt == 'Изображение' and rv.value_id:
                        iv = session.query(ImageValue).get(rv.value_id)
                        content = {'url': iv.url, 'file_path': iv.file_path, 'format': iv.format} if iv else None
                    elif mt == 'Геоданные' and rv.value_id:
                        gv = session.query(GeodataValue).get(rv.value_id)
                        if gv:
                            from geoalchemy2.shape import to_shape
                            geom = to_shape(gv.geometry)
                            content = {'geojson': geom.__geo_interface__, 'type': gv.geometry_type}
                
                author_name = None
                source_name = None
                if r.resource_static and r.resource_static.bibliographic:
                    if r.resource_static.bibliographic.author:
                        author_name = r.resource_static.bibliographic.author.name
                    if r.resource_static.bibliographic.source:
                        source_name = r.resource_static.bibliographic.source.name
                
                result.append(ResourceResult(
                    id=r.id, title=r.title, uri=r.uri,
                    author=author_name,
                    source=source_name,
                    modality_type=rv.modality.modality_type if rv and rv.modality else None,
                    content=content,
                    features=r.features
                ))
            return result