import os
import json
import logging
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class GeoDataImporter:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        self.conn = None

    def connect(self):
        """Установка соединения с БД"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            logger.info("Успешное подключение к базе данных")
            return True
        except psycopg2.Error as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            return False
        
    def simplify_geo_name(self, geo_name):
        """Упрощает название географического объекта, убирая указания районов"""
        if not geo_name:
            return geo_name
        
        if ',' in geo_name:
            parts = [part.strip() for part in geo_name.split(',')]
            return parts[-1]
        
        return geo_name.strip()

    def import_geojson(self, file_path: str):
        """Импорт данных из GeoJSON файла"""
        if not self.connect():
            return
        
        try:
            # Чтение GeoJSON файла
            with open(file_path, 'r', encoding='utf-8') as f:
                geodata = json.load(f)
            
            with self.conn.cursor() as cursor:
                # Обработка каждого объекта в GeoJSON
                for name, feature in geodata.items():
                    logger.info(f"Обработка объекта: {name}")
                    
                    # 1. Создаем географическую сущность для основного названия
                    geo_entity_id = self.get_or_create_geographical_entity(cursor, name)
                    
                    # 2. Создаем контент карты
                    map_content_id = self.create_map_content(cursor, name, feature['geometry'])
                    
                    # 3. Создаем связь
                    self.create_geo_relation(cursor, map_content_id, geo_entity_id)
                    
                    # 4. Дополнительно: создаем связи для упрощенных названий
                    simplified_name = self.simplify_geo_name(name)
                    if simplified_name != name:
                        simplified_geo_id = self.get_or_create_geographical_entity(cursor, simplified_name)
                        self.create_geo_relation(cursor, map_content_id, simplified_geo_id)
                
                self.conn.commit()
                logger.info(f"Успешно импортировано {len(geodata)} объектов")
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Ошибка импорта: {e}")
        finally:
            if self.conn:
                self.conn.close()

    def get_or_create_geographical_entity(self, cursor, name_ru: str) -> int:
        """Проверяет существование географической сущности и создает при необходимости"""
        try:
            # Проверка существования объекта
            cursor.execute(
                "SELECT id FROM geographical_entity WHERE name_ru = %s",
                (name_ru,)
            )
            result = cursor.fetchone()
            
            if result:
                logger.info(f"Географический объект '{name_ru}' уже существует (id={result[0]})")
                return result[0]
            
            # Создание нового объекта
            cursor.execute(
                "INSERT INTO geographical_entity (name_ru) VALUES (%s) RETURNING id",
                (name_ru,)
            )
            new_id = cursor.fetchone()[0]
            logger.info(f"Создан новый географический объект: '{name_ru}' (id={new_id})")
            return new_id
            
        except psycopg2.Error as e:
            logger.error(f"Ошибка работы с географическими объектами: {e}")
            raise

    def create_geo_relation(self, cursor, map_content_id: int, geo_entity_id: int):
        """Создает связь между контентом карты и географической сущностью"""
        try:
            cursor.execute(
                """
                INSERT INTO entity_geo (entity_id, entity_type, geographical_entity_id)
                VALUES (%s, 'map_content', %s)
                """,
                (map_content_id, geo_entity_id)
            )
            logger.info(f"Создана связь: map_content={map_content_id} ↔ geo_entity={geo_entity_id}")
            
        except psycopg2.Error as e:
            logger.error(f"Ошибка создания связи: {e}")
            raise
    def create_map_content(self, cursor, title: str, geometry: dict) -> int:
        """Создает запись в map_content на основе GeoJSON геометрии, если она не существует"""
        try:
            # Проверяем, существует ли уже такая геометрия
            cursor.execute(
                """
                SELECT id FROM map_content 
                WHERE ST_Equals(geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                LIMIT 1
                """,
                (json.dumps(geometry),)
            )
            result = cursor.fetchone()
            
            if result:
                logger.info(f"Геометрия для '{title}' уже существует (id={result[0]})")
                return result[0]
            
            # Создаем новую запись, если геометрия не существует
            cursor.execute(
                """
                INSERT INTO map_content (title, geometry)
                VALUES (%s, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                RETURNING id
                """,
                (title, json.dumps(geometry))
            )
            new_id = cursor.fetchone()[0]
            logger.info(f"Создан новый контент карты: '{title}' (id={new_id})")
            return new_id
            
        except psycopg2.Error as e:
            logger.error(f"Ошибка создания контента карты: {e}")
            raise
if __name__ == "__main__":
    importer = GeoDataImporter()
    importer.import_geojson("/var/www/salut_bot/json_files/geodb.json")