# generate_real_embeddings.py
import os
import sys
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json
import json
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Добавляем путь к корневой директории проекта
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embedding_config import embedding_config, get_model_dimension

class RealEmbeddingGenerator:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "eco"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "Fdf78yh0a4b!"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
        
        current_model = os.getenv("EMBEDDING_MODEL", embedding_config.current_model)
        embedding_dimension = os.getenv("EMBEDDING_DIMENSION")
        
        if embedding_dimension:
            self.embedding_dimension = int(embedding_dimension)
        else:
            self.embedding_dimension = get_model_dimension(current_model)
            
        current_dir = Path(__file__).parent
        base_dir = current_dir.parent.parent
        embedding_models_dir = base_dir / "embedding_models" / "BERTA"
        
        self.embedding_model_path = str(embedding_models_dir)
        
        print(f"📏 Размерность эмбеддингов: {self.embedding_dimension}")
        print(f"🎯 Активная модель: {current_model}")
        print(f"📁 Путь к модели: {self.embedding_model_path}")
        
        self.conn = None
        self.cur = None
        self.embedding_model = None

    def connect(self):
        """Установка соединения с базой данных"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            print("✅ Успешное подключение к базе данных")
        except Exception as e:
            print(f"❌ Ошибка подключения к базе данных: {e}")
            raise

    def disconnect(self):
        """Закрытие соединения с базой данных"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("🔌 Соединение с базой данных закрыто")

    def load_embedding_model(self):
        """Загрузка модели для генерации эмбеддингов"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            
            # Проверяем, что модель работает
            test_embedding = embeddings.embed_query("test")
            if test_embedding is None or len(test_embedding) == 0:
                raise Exception("Model loaded but returned empty embedding")
                
            print(f"✅ Модель эмбеддингов успешно загружена, размерность: {len(test_embedding)}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели эмбеддингов: {e}")
            return None

    def generate_embedding(self, text):
        """Генерация эмбеддинга для текста"""
        if not text:
            print("⚠️  Пустой текст для эмбеддинга")
            return None
        
        if not self.embedding_model:
            print("❌ Модель эмбеддингов не загружена")
            return None
        
        try:
            embedding = self.embedding_model.embed_query(text)
            
            if embedding is None:
                print("❌ Модель вернула None")
                return None
                
            if len(embedding) != self.embedding_dimension:
                print(f"⚠️  Предупреждение: Размерность эмбеддинга ({len(embedding)}) не совпадает с ожидаемой ({self.embedding_dimension})")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Ошибка генерации эмбеддинга: {e}")
            return None

    def is_zero_embedding(self, embedding_data):
        """Проверяет, является ли эмбеддинг нулевым вектором"""
        try:
            if isinstance(embedding_data, str):
                # Это строка в формате вектора
                if embedding_data.startswith('[') and embedding_data.endswith(']'):
                    numbers_str = embedding_data[1:-1]
                    numbers = [float(x.strip()) for x in numbers_str.split(',') if x.strip()]
                    arr = np.array(numbers, dtype=float)
                else:
                    return False
            else:
                # Уже массив
                arr = np.array(embedding_data, dtype=float)
            
            return np.allclose(arr, 0.0, atol=1e-10)
            
        except Exception as e:
            print(f"⚠️  Ошибка проверки нулевого эмбеддинга: {e}")
            return False

    def get_text_for_embedding(self, text_content):
        """Собирает текст для генерации эмбеддинга из всех доступных полей"""
        text_parts = []
        
        # Добавляем заголовок
        if text_content.get('title'):
            text_parts.append(text_content['title'])
        
        # Добавляем контент
        if text_content.get('content'):
            text_parts.append(text_content['content'])
        
        # Добавляем описание
        if text_content.get('description'):
            text_parts.append(text_content['description'])
        
        # Обрабатываем structured_data
        if text_content.get('structured_data'):
            structured_data = text_content['structured_data']
            
            # Рекурсивно собираем все строковые значения из structured_data
            def extract_text_values(data):
                if isinstance(data, dict):
                    return ' '.join(extract_text_values(value) for value in data.values())
                elif isinstance(data, list):
                    return ' '.join(extract_text_values(item) for item in data)
                elif isinstance(data, str):
                    return data
                else:
                    return ''
            
            structured_text = extract_text_values(structured_data).strip()
            if structured_text:
                text_parts.append(structured_text)
        
        # Объединяем все части
        combined_text = ' '.join(text_parts).strip()
        
        if not combined_text:
            print("⚠️  Не удалось собрать текст для генерации эмбеддинга")
            return None
            
        print(f"📝 Текст для эмбеддинга: {combined_text[:200]}...")
        return combined_text

    def get_all_text_contents(self):
        """Получает все text_content"""
        try:
            query = """
            SELECT id, title, content, description, structured_data, embedding 
            FROM text_content 
            WHERE embedding IS NOT NULL
            ORDER BY id
            """
            
            self.cur.execute(query)
            text_contents = []
            
            for row in self.cur.fetchall():
                text_id, title, content, description, structured_data, embedding = row
                
                text_contents.append({
                    'id': text_id,
                    'title': title,
                    'content': content,
                    'description': description,
                    'structured_data': structured_data,
                    'embedding': embedding
                })
            
            return text_contents
            
        except Exception as e:
            print(f"❌ Ошибка получения text_content: {e}")
            return []

    def update_embedding(self, text_id, new_embedding):
        """Обновляет эмбеддинг для указанного text_content"""
        try:
            # Форматируем эмбеддинг как вектор
            vector_str = '[' + ','.join(str(x) for x in new_embedding) + ']'
            
            self.cur.execute(
                "UPDATE text_content SET embedding = %s::vector WHERE id = %s",
                (vector_str, text_id)
            )
            return True
        except Exception as e:
            print(f"❌ Ошибка обновления эмбеддинга для text_content {text_id}: {e}")
            return False

    def generate_real_embeddings(self):
        """Генерирует настоящие эмбеддинги для всех нулевых записей"""
        print("🔄 Получение всех text_content...")
        
        text_contents = self.get_all_text_contents()
        
        if not text_contents:
            print("❌ Не найдено text_content в базе данных")
            return
        
        print(f"📊 Найдено {len(text_contents)} записей с эмбеддингами")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        zero_embedding_count = 0
        
        # Сначала подсчитаем нулевые эмбеддинги
        for text_content in text_contents:
            if self.is_zero_embedding(text_content['embedding']):
                zero_embedding_count += 1
        
        print(f"🔍 Найдено {zero_embedding_count} нулевых эмбеддингов из {len(text_contents)}")
        
        if zero_embedding_count == 0:
            print("✅ Все эмбеддинги уже являются настоящими, обновление не требуется")
            return
        
        # Обновляем только нулевые эмбеддинги
        for i, text_content in enumerate(text_contents, 1):
            text_id = text_content['id']
            title = text_content['title'] or 'Без названия'
            
            # Пропускаем если эмбеддинг не нулевой
            if not self.is_zero_embedding(text_content['embedding']):
                continue
                
            print(f"\n🔧 Обработка {i}/{len(text_contents)}: ID {text_id} - '{title}'")
            
            try:
                # Собираем текст для генерации эмбеддинга
                text_for_embedding = self.get_text_for_embedding(text_content)
                
                if not text_for_embedding:
                    print(f"⚠️  Пропуск записи {text_id}: не удалось собрать текст")
                    skipped_count += 1
                    continue
                
                # Генерируем новый эмбеддинг
                new_embedding = self.generate_embedding(text_for_embedding)
                
                if new_embedding is None:
                    print(f"❌ Не удалось сгенерировать эмбеддинг для записи {text_id}")
                    error_count += 1
                    continue
                
                # Проверяем, что новый эмбеддинг не нулевой
                if self.is_zero_embedding(new_embedding):
                    print(f"⚠️  Сгенерированный эмбеддинг тоже нулевой для записи {text_id}")
                    error_count += 1
                    continue
                
                # Обновляем эмбеддинг в базе данных
                if self.update_embedding(text_id, new_embedding):
                    print(f"✅ Обновлен эмбеддинг для записи {text_id}")
                    success_count += 1
                else:
                    error_count += 1
                    
                # Коммитим каждые 10 записей для надежности
                if success_count % 10 == 0:
                    self.conn.commit()
                    print(f"💾 Выполнен коммит после {success_count} успешных обновлений")
                    
            except Exception as e:
                print(f"❌ Ошибка обработки записи {text_id}: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1
        
        # Финальный коммит
        self.conn.commit()
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ ЭМБЕДДИНГОВ:")
        print(f"   ✅ Успешно обновлено: {success_count}")
        print(f"   ❌ Ошибок: {error_count}")
        print(f"   ⚠️  Пропущено: {skipped_count}")
        print(f"   🔧 Всего нулевых эмбеддингов: {zero_embedding_count}")
        print(f"   📝 Всего обработано записей: {len(text_contents)}")

    def run(self):
        """Запуск процесса генерации эмбеддингов"""
        try:
            self.connect()
            
            # Загружаем модель эмбеддингов
            print("🔄 Загрузка модели эмбеддингов...")
            self.embedding_model = self.load_embedding_model()
            
            if not self.embedding_model:
                print("❌ Не удалось загрузить модель эмбеддингов. Процесс остановлен.")
                return
            
            # Запускаем генерацию
            self.generate_real_embeddings()
            
        except Exception as e:
            print(f"❌ Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()

def main():
    """Основная функция"""
    print("🚀 Запуск генерации настоящих эмбеддингов...")
    print("=" * 50)
    
    generator = RealEmbeddingGenerator()
    generator.run()
    
    print("=" * 50)
    print("🏁 Процесс завершен")

if __name__ == "__main__":
    main()