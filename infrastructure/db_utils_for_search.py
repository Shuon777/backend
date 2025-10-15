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

        # Запрос для получения текущей "страницы" результатов
        query = """
        SELECT DISTINCT title FROM text_content
        WHERE 
            title ~* %s
            AND (content IS NOT NULL AND content != '' OR structured_data IS NOT NULL AND structured_data::text != '{}'::text)
        ORDER BY title
        LIMIT %s OFFSET %s;
        """
        
        # Запрос для проверки, есть ли еще результаты (проверяем наличие хотя бы одной записи после текущего offset+limit)
        check_more_query = """
        SELECT 1 FROM text_content
        WHERE title ~* %s
        AND (content IS NOT NULL AND content != '' OR structured_data IS NOT NULL AND structured_data::text != '{}'::text)
        OFFSET %s LIMIT 1;
        """

        try:
            search_pattern = rf'\y{object_name.lower()}\y'
            
            # Выполняем основной запрос
            cursor.execute(query, (search_pattern, limit, offset))
            results = cursor.fetchall()
            matches = [row['title'] for row in results]

            # Выполняем запрос для проверки наличия следующих страниц
            cursor.execute(check_more_query, (search_pattern, offset + limit))
            has_more = cursor.fetchone() is not None

            if not matches:
                return {"status": "not_found", "matches": [], "has_more": False}
            elif len(matches) == 1:
                return {"status": "found", "matches": matches, "has_more": False}
            else:
                return {"status": "ambiguous", "matches": matches, "has_more": has_more}

        finally:
            conn.close()