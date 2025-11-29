import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

class Slot_validator:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
    
    def is_known_object(self, object_name: str) -> dict:

        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
        SELECT name_ru FROM resource_identifiers
        WHERE LOWER(name_ru) LIKE %s
        """
        cursor.execute(query, (f'%{object_name.lower()}%',))
        
        results = cursor.fetchall()
        conn.close()

        matches = [row['name_ru'] for row in results]

        if not matches:
            return {"known": False, "matches": []}
        elif len(matches) == 1:
            return {"known": True, "matches": matches}
        else:
            return {"known": "ambiguous", "matches": matches}
        
    def find_species_with_description(self, object_name: str, limit: int = 5, offset: int = 0) -> dict:
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # ИСПРАВЛЕННЫЙ запрос - ищем в biological_entity по русским и научным названиям
        query = """
        SELECT DISTINCT be.common_name_ru as title 
        FROM biological_entity be
        JOIN entity_relation er ON be.id = er.target_id 
            AND er.target_type = 'biological_entity'
            AND er.relation_type = 'описание объекта'
        JOIN text_content tc ON tc.id = er.source_id 
            AND er.source_type = 'text_content'
        WHERE 
            (be.common_name_ru ~* %s OR be.scientific_name ~* %s)
            AND (tc.content IS NOT NULL AND tc.content != '' OR tc.structured_data IS NOT NULL AND tc.structured_data::text != '{}'::text)
        ORDER BY title
        LIMIT %s OFFSET %s;
        """
        
        # Запрос для проверки, есть ли еще результаты
        check_more_query = """
        SELECT 1 
        FROM biological_entity be
        JOIN entity_relation er ON be.id = er.target_id 
            AND er.target_type = 'biological_entity'
            AND er.relation_type = 'описание объекта'
        JOIN text_content tc ON tc.id = er.source_id 
            AND er.source_type = 'text_content'
        WHERE 
            (be.common_name_ru ~* %s OR be.scientific_name ~* %s)
            AND (tc.content IS NOT NULL AND tc.content != '' OR tc.structured_data IS NOT NULL AND tc.structured_data::text != '{}'::text)
        OFFSET %s LIMIT 1;
        """

        try:
            search_pattern = rf'\y{object_name.lower()}\y'
            
            # Выполняем основной запрос
            cursor.execute(query, (search_pattern, search_pattern, limit, offset))
            results = cursor.fetchall()
            matches = [row['title'] for row in results]

            # Выполняем запрос для проверки наличия следующих страниц
            cursor.execute(check_more_query, (search_pattern, search_pattern, offset + limit))
            has_more = cursor.fetchone() is not None

            # СОХРАНЯЕМ ОРИГИНАЛЬНУЮ ЛОГИКУ СТАТУСОВ
            if not matches:
                return {"status": "not_found", "matches": [], "has_more": False}
            elif len(matches) == 1:
                return {"status": "found", "matches": matches, "has_more": has_more}
            else:
                return {"status": "ambiguous", "matches": matches, "has_more": has_more}

        except Exception as e:
            return {"status": "error", "message": str(e), "matches": []}
        finally:
            conn.close()