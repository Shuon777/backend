from typing import List
from ..use_cases.interfaces import ObjectPropertyRepository
from .database_client import DatabaseClient

class PostgresObjectPropertyRepository(ObjectPropertyRepository):
    def __init__(self, client: DatabaseClient):
        self._client = client

    def add_or_update_property(self, object_type_id: int, property_name: str, values: List[str]) -> None:
        if not values:
            return
        sql = """
            INSERT INTO eco_assistant.object_property (object_type_id, property_name, property_values)
            VALUES (%s, %s, %s)
            ON CONFLICT (object_type_id, property_name)
            DO UPDATE SET
                property_values = array(
                    SELECT DISTINCT unnest(eco_assistant.object_property.property_values || %s)
                ),
                updated_at = now()
        """
        self._client.execute(sql, (object_type_id, property_name.lower(), values, values))
        self._client.commit()