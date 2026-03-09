# -*- coding: utf-8 -*-
"""Скрипт для индексации текстовых ресурсов в FAISS векторную базу"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# Добавляем путь к проекту для импорта конфигурации
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from embedding_config import embedding_config, get_model_dimension
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.vectorstores.utils import DistanceStrategy

class TextResourceIndexer:
    def __init__(self, use_local_model: bool = True):
        """
        Инициализация индексатора
        
        Args:
            use_local_model: Использовать локальную модель (True) или загружать из HuggingFace (False)
        """
        self.use_local_model = use_local_model
        self.embedding_model = None
        
        # Получаем конфигурацию модели
        current_model = embedding_config.current_model
        self.model_name = current_model
        
        if use_local_model:
            # Используем локальный путь к модели
            self.embedding_model_path = embedding_config.get_model_path(current_model)
            print(f"Используем локальную модель: {self.embedding_model_path}")
        else:
            # Используем имя модели из HuggingFace
            self.embedding_model_path = current_model
            print(f"Используем модель из HuggingFace: {self.embedding_model_path}")
        
    def load_embedding_model(self):
        """Загрузка модели для эмбеддингов"""
        print(f"Загружаем модель эмбеддингов: {self.embedding_model_path}")
        
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Тестируем модель
            test_embedding = self.embedding_model.embed_query("тест")
            print(f"Модель загружена успешно, размерность эмбеддингов: {len(test_embedding)}")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            
            # Пробуем альтернативный путь для локальной модели
            if self.use_local_model:
                print("Пробуем использовать путь /embedding_models/BERTA")
                try:
                    # Ищем модель в стандартной директории
                    base_dir = Path(__file__).parent.parent.parent
                    model_path = base_dir / "embedding_models" / "BERTA"
                    
                    if model_path.exists():
                        self.embedding_model = HuggingFaceEmbeddings(
                            model_name=str(model_path),
                            model_kwargs={'device': 'cpu'},
                            encode_kwargs={'normalize_embeddings': False}
                        )
                        print(f"Модель загружена из альтернативного пути: {model_path}")
                    else:
                        raise FileNotFoundError(f"Директория модели не найдена: {model_path}")
                except Exception as e2:
                    print(f"Не удалось загрузить локальную модель: {e2}")
                    print("Пробуем загрузить из HuggingFace как резервный вариант...")
                    
                    # Резервный вариант: загружаем из HuggingFace
                    self.embedding_model = HuggingFaceEmbeddings(
                        model_name="sergeyzh/BERTA",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': False}
                    )
                    print("Модель загружена из HuggingFace (резервный вариант)")
    
    def split_into_chunks(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """
        Разбиение текста на чанки с гарантией максимального размера
        
        Args:
            text: Исходный текст
            max_chunk_size: Максимальный размер чанка в символах
            
        Returns:
            Список чанков
        """
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        
        # Если текст короче максимального размера, возвращаем как есть
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        
        # Сначала пробуем разбить по абзацам
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            # Если абзац меньше max_chunk_size, добавляем как есть
            if len(paragraph) <= max_chunk_size:
                chunks.append(paragraph)
                continue
            
            # Разбиваем абзац на предложения
            sentences = self._split_into_sentences(paragraph)
            current_chunk = ""
            
            for sentence in sentences:
                # Если одно предложение больше max_chunk_size
                if len(sentence) > max_chunk_size:
                    # Сначала сохраняем текущий чанк если он не пустой
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Разбиваем очень длинное предложение на части с overlap
                    parts = self._split_long_sentence(sentence, max_chunk_size)
                    chunks.extend(parts)
                    continue
                
                # Если добавление предложения превысит лимит, сохраняем текущий чанк
                if len(current_chunk) + len(sentence) + 2 > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Добавляем последний чанк из абзаца
            if current_chunk:
                chunks.append(current_chunk)
        
        # ВТОРАЯ ЗАЩИТА: проверяем, что все чанки не превышают max_chunk_size
        final_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Если чанк все еще слишком длинный, принудительно разбиваем
                print(f"⚠️ ВНИМАНИЕ: Чанк длиной {len(chunk)} символов все еще превышает лимит!")
                print(f"   Начало чанка: {chunk[:100]}...")
                parts = self._split_long_sentence(chunk, max_chunk_size)
                final_chunks.extend(parts)
        
        return final_chunks

    def _split_long_sentence(self, sentence: str, max_chunk_size: int = 512, overlap: int = 128) -> List[str]:
        """
        Разбивает очень длинное предложение на части с overlap
        
        Args:
            sentence: Очень длинное предложение
            max_chunk_size: Максимальный размер части
            overlap: Перекрытие между частями
            
        Returns:
            Список частей предложения
        """
        if len(sentence) <= max_chunk_size:
            return [sentence]
        
        parts = []
        start = 0
        sentence_length = len(sentence)
        
        while start < sentence_length:
            # Определяем конец текущей части
            end = start + max_chunk_size
            
            # Если это не последняя часть, ищем хорошее место для разбиения
            if end < sentence_length:
                # Ищем ближайший пробел или знак препинания
                for i in range(end, max(start, end - 100), -1):
                    if i < sentence_length and sentence[i] in ' ,.;:!?—':
                        end = i + 1  # Включаем разделитель
                        break
                else:
                    # Если не нашли хорошее место, просто режем
                    end = start + max_chunk_size
            
            # Добавляем часть
            part = sentence[start:end].strip()
            if part:
                parts.append(part)
            
            # Двигаемся дальше с overlap
            start = end - overlap
            if start < 0:
                start = 0
            # Если мы уже в конце или почти в конце
            if start >= sentence_length or end >= sentence_length:
                break
        
        # Добавляем последний кусок если остался
        if start < sentence_length:
            last_part = sentence[start:].strip()
            if last_part:
                parts.append(last_part)
        
        print(f"   Разбито на {len(parts)} частей")
        return parts
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Простое разбиение на предложения
        
        Args:
            text: Исходный текст
            
        Returns:
            Список предложений
        """
        # Простое разбиение по точкам, восклицательным и вопросительным знакам
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_text_from_resource(self, resource: Dict[str, Any]) -> str:
        """
        Извлекает текст из ресурса для индексации
        
        Args:
            resource: Ресурс из JSON
            
        Returns:
            Объединенный текст для индексации
        """
        parts = []
        
        # Всегда добавляем common name из идентификатора
        identificator = resource.get('identificator', {})
        name_info = identificator.get('name', {})
        common_name = name_info.get('common', '')
        
        if common_name:
            parts.append(common_name)
        
        # Для текстовых ресурсов добавляем content
        if resource.get('type') == 'Текст':
            content = resource.get('content', '')
            if content:
                parts.append(content)
        
        # Для географических объектов добавляем description
        elif resource.get('type') == 'Географический объект':
            description = resource.get('description', '')
            if description:
                parts.append(description)
        
        # Объединяем всё: Название. Описание
        if len(parts) == 2:
            # Если есть и название и описание, соединяем их через точку
            return f"{parts[0]}. {parts[1]}"
        elif len(parts) == 1:
            # Если только название или только описание
            return parts[0]
        else:
            return ""
    
    def process_resources(self, json_file_path: str) -> List[Document]:
        """
        Обрабатывает ресурсы из JSON файла и создает документы
        
        Args:
            json_file_path: Путь к JSON файлу
            
        Returns:
            Список документов для индексации
        """
        print(f"Читаем файл: {json_file_path}")
        very_long_chunks = []  
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения JSON файла: {e}")
            return []
        
        resources = data.get('resources', [])
        print(f"Найдено {len(resources)} ресурсов")
        
        text_count = 0
        geo_count = 0
        documents = []
        
        for i, resource in enumerate(resources):
            resource_type = resource.get('type', 'Неизвестно')
            
            # Фильтруем только текстовые ресурсы
            if resource_type == 'Текст':
                text_count += 1
            elif resource_type == 'Географический объект':
                geo_count += 1
            else:
                continue
            
            # Извлекаем текст для индексации
            full_text = self.extract_text_from_resource(resource)
            
            if not full_text.strip():
                continue
            
            # Разбиваем на чанки
            chunks = self.split_into_chunks(full_text)
            for chunk in chunks:
                if len(chunk) > 512:
                    very_long_chunks.append({
                        "resource": common_name,
                        "chunk_length": len(chunk),
                        "preview": chunk[:100] + "..."
                    })
                
            # Логируем для отладки
            identificator = resource.get('identificator', {})
            name_info = identificator.get('name', {})
            common_name = name_info.get('common', 'unknown')
            
            if len(chunks) > 1:
                print(f"Ресурс '{common_name}': разделен на {len(chunks)} чанков")
            
            # Создаем документы для каждого чанка
            for chunk_idx, chunk_text in enumerate(chunks):
                # Получаем метаданные
                identificator = resource.get('identificator', {})
                name_info = identificator.get('name', {})
                
                metadata = {
                    "resource_id": identificator.get('id', f'unknown_{i}'),
                    "resource_type": resource_type,
                    "source": name_info.get('source', ''),
                    "common_name": name_info.get('common', ''),
                    "scientific_name": name_info.get('scientific', ''),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "original_length": len(chunk_text),
                    "full_text": full_text[:500] + "..." if len(full_text) > 500 else full_text  # Сохраняем для отладки
                }
                
                # Создаем документ
                doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                
                documents.append(doc)
            
            if (i + 1) % 100 == 0:
                print(f"Обработано {i + 1} ресурсов, создано {len(documents)} чанков")
        
        print(f"\nСтатистика обработки:")
        print(f"  Текстовых ресурсов: {text_count}")
        print(f"  Географических объектов: {geo_count}")
        print(f"  Всего создано чанков: {len(documents)}")
        
        # Анализируем размеры чанков
        chunk_sizes = [len(doc.page_content) for doc in documents]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            print(f"  Средний размер чанка: {avg_size:.1f} символов")
            print(f"  Минимальный размер: {min_size} символов")
            print(f"  Максимальный размер: {max_size} символов")
            
            # Подсчет чанков по размеру
            small = sum(1 for s in chunk_sizes if s < 100)
            good = sum(1 for s in chunk_sizes if 100 <= s <= 512)
            large = sum(1 for s in chunk_sizes if s > 512)
            print(f"  Чанков < 100 символов: {small}")
            print(f"  Чанков 100-512 символов: {good}")
            print(f"  Чанков > 512 символов: {large}")
        if very_long_chunks:
            print(f"\n⚠️ ВНИМАНИЕ: Найдено {len(very_long_chunks)} чанков длиннее 512 символов!")
            print("Примеры:")
            for i, info in enumerate(very_long_chunks[:5]):
                print(f"  {i+1}. {info['resource']}: {info['chunk_length']} символов")
                print(f"     {info['preview']}")
        return documents
    
    def create_faiss_index(self, documents: List[Document], output_dir: str):
        """
        Создает FAISS индекс из документов
        
        Args:
            documents: Список документов
            output_dir: Директория для сохранения индекса
        """
        if not self.embedding_model:
            self.load_embedding_model()
        
        if not self.embedding_model:
            print("❌ Не удалось загрузить модель эмбеддингов")
            return None
        
        print("\nСоздаем FAISS индекс...")
        
        try:
            # Создаем векторное хранилище
            vectorstore = LangchainFAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            
            # Сохраняем индекс
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            vectorstore.save_local(str(output_path))
            
            print(f"✅ FAISS индекс сохранен в {output_path}")
            print(f"📊 Размер индекса: {len(documents)} документов")
            
            # Сохраняем статистику
            stats_file = output_path / "index_stats.json"
            stats = {
                "total_documents": len(documents),
                "embedding_model": self.model_name,
                "embedding_model_path": str(self.embedding_model_path) if hasattr(self, 'embedding_model_path') else self.model_name,
                "use_local_model": self.use_local_model,
                "chunk_size_limit": 512,
                "index_created": True,
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"📝 Статистика сохранена в {stats_file}")
            
            return vectorstore
            
        except Exception as e:
            print(f"❌ Ошибка создания FAISS индекса: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_faiss_index(self, index_dir: str):
        """
        Загружает существующий FAISS индекс
        
        Args:
            index_dir: Директория с индексом
            
        Returns:
            Загруженное векторное хранилище
        """
        if not self.embedding_model:
            self.load_embedding_model()
        
        if not self.embedding_model:
            print("❌ Не удалось загрузить модель эмбеддингов")
            return None
        
        print(f"Загружаем FAISS индекс из {index_dir}")
        
        try:
            vectorstore = LangchainFAISS.load_local(
                index_dir,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            print(f"✅ Индекс загружен, размер: {vectorstore.index.ntotal} векторов")
            return vectorstore
            
        except Exception as e:
            print(f"❌ Ошибка загрузки индекса: {e}")
            return None
    
    def search_similar(self, vectorstore, query: str, k: int = 5) -> List[Document]:
        """
        Ищет похожие документы
        
        Args:
            vectorstore: Векторное хранилище
            query: Поисковый запрос
            k: Количество результатов
            
        Returns:
            Список похожих документов
        """
        if not vectorstore:
            print("❌ Векторное хранилище не инициализировано")
            return []
        
        try:
            results = vectorstore.similarity_search(query, k=k)
            
            print(f"\nРезультаты поиска для запроса: '{query}'")
            print("=" * 60)
            
            for i, doc in enumerate(results):
                print(f"\nРезультат #{i + 1}:")
                print(f"📋 Тип: {doc.metadata.get('resource_type', 'Неизвестно')}")
                print(f"🏷️  Название: {doc.metadata.get('common_name', 'Без названия')}")
                if doc.metadata.get('scientific_name'):
                    print(f"🔬 Научное название: {doc.metadata.get('scientific_name')}")
                print(f"📊 Чанк: {doc.metadata.get('chunk_index', 0) + 1}/{doc.metadata.get('total_chunks', 1)}")
                print(f"📝 Контент: {doc.page_content[:200]}...")
                print(f"🔗 ID ресурса: {doc.metadata.get('resource_id', 'Неизвестно')}")
                print("-" * 60)
            
            return results
            
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
            return []
    
    def analyze_index(self, vectorstore):
        """
        Анализирует содержимое индекса
        
        Args:
            vectorstore: Векторное хранилище
        """
        if not vectorstore:
            print("❌ Векторное хранилище не инициализировано")
            return
        
        print("\n" + "=" * 60)
        print("АНАЛИЗ ИНДЕКСА")
        print("=" * 60)
        
        # Получаем все документы (ограниченное количество для анализа)
        sample_queries = ["байкал", "растение", "животное", "озеро"]
        
        for query in sample_queries:
            results = vectorstore.similarity_search(query, k=3)
            
            print(f"\nЗапрос: '{query}'")
            print(f"Найдено результатов: {len(results)}")
            
            resource_types = {}
            for doc in results:
                rtype = doc.metadata.get('resource_type', 'unknown')
                resource_types[rtype] = resource_types.get(rtype, 0) + 1
            
            for rtype, count in resource_types.items():
                print(f"  {rtype}: {count}")
        
        print("\n" + "=" * 60)


def main():
    """Основная функция для индексации текстов"""
    # Конфигурация
    BASE_DIR = Path(__file__).parent.parent.parent
    JSON_FILE = BASE_DIR / "json_files" / "resources_dist.json"
    INDEX_DIR = BASE_DIR / "knowledge_base_scripts" / "Vector" / "faiss_index"
    
    print("=" * 60)
    print("FAISS ИНДЕКСАЦИЯ ТЕКСТОВЫХ РЕСУРСОВ")
    print("=" * 60)
    print("Настройки разбиения на чанки:")
    print("  - Максимальный размер чанка: 512 символов")
    print("  - Сначала по абзацам, потом по предложениям")
    print("  - Большие предложения (>512) остаются как отдельные чанки")
    print("=" * 60)
    
    # Проверяем существование файла
    if not JSON_FILE.exists():
        print(f"❌ Файл не найден: {JSON_FILE}")
        print("Проверьте путь к файлу resources_dist.json")
        
        # Пробуем альтернативные пути
        alt_paths = [
            BASE_DIR.parent / "json_files" / "resources_dist.json",
            Path("/json_files") / "resources_dist.json",
            Path.cwd() / "json_files" / "resources_dist.json"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                JSON_FILE = alt_path
                print(f"✅ Найден альтернативный путь: {JSON_FILE}")
                break
        
        if not JSON_FILE.exists():
            print("❌ Не удалось найти файл resources_dist.json")
            return
    
    print(f"📄 JSON файл: {JSON_FILE}")
    print(f"📁 Директория индекса: {INDEX_DIR}")
    
    # Создаем индексатор с использованием локальной модели
    indexer = TextResourceIndexer(use_local_model=True)
    
    # Обрабатываем ресурсы
    documents = indexer.process_resources(str(JSON_FILE))
    
    if not documents:
        print("❌ Не найдено документов для индексации")
        return
    
    # Создаем FAISS индекс
    vectorstore = indexer.create_faiss_index(documents, str(INDEX_DIR))
    
    if not vectorstore:
        print("❌ Не удалось создать индекс")
        return
    
    # Анализируем индекс
    indexer.analyze_index(vectorstore)
    
    # Тестовый поиск
    print("\n" + "=" * 60)
    print("ТЕСТОВЫЙ ПОИСК")
    print("=" * 60)
    
    test_queries = [
        "байкальская нерпа",
        "земляника лесная",
        "археологические находки Шалино",
        "растения Байкала",
        "животные озера Байкал"
    ]
    
    for query in test_queries:
        indexer.search_similar(vectorstore, query, k=3)
    
    print("\n" + "=" * 60)
    print("✅ ИНДЕКСАЦИЯ УСПЕШНО ЗАВЕРШЕНА")
    print(f"📁 Индекс сохранен в: {INDEX_DIR}")
    print("=" * 60)
    
    # Проверяем, что файлы созданы
    index_files = list(Path(INDEX_DIR).glob("*"))
    if index_files:
        print("\nСозданные файлы индекса:")
        for file in index_files:
            size_kb = file.stat().st_size / 1024
            print(f"  {file.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()