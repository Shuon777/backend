from typing import List
from ..use_cases.interfaces import ResourceFeatureRepository
from .database_client import DatabaseClient

class PostgresResourceFeatureRepository(ResourceFeatureRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client

    def add_or_update_feature(self, modality_id: int, feature_name: str, values: List[str]) -> None:
        if not values:
            return
        sql = """
            INSERT INTO eco_assistant.resource_feature (modality_id, feature_name, feature_values)
            VALUES (%s, %s, %s)
            ON CONFLICT (modality_id, feature_name)
            DO UPDATE SET
                feature_values = array(
                    SELECT DISTINCT unnest(eco_assistant.resource_feature.feature_values || %s)
                ),
                updated_at = now()
        """
        self._client.execute(sql, (modality_id, feature_name.lower(), values, values))
        self._client.commit()