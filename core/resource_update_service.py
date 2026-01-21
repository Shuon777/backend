import json
import os
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class ResourceUpdateService:
    def __init__(self, resources_dist_path: str, images_dir: str):
        self.resources_dist_path = resources_dist_path
        self.images_dir = images_dir
        self.temp_dir = None
        
    def extract_archive_chunked(self, archive_path: str, extract_to: str, chunk_size_mb: int = 100) -> Dict:
        """Распаковывает архив чанками и возвращает статистику"""
        results = {
            "total_files": 0,
            "extracted_files": 0,
            "chunks": 0,
            "errors": [],
            "file_list": []
        }
        
        try:
            print(f"Распаковка архива {archive_path} в {extract_to}")
            
            # Создаем папку для распаковки если не существует
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Получаем список файлов в архиве
                file_list = zip_ref.namelist()
                results["total_files"] = len(file_list)
                print(f"Файлов в архиве: {len(file_list)}")
                
                # Разбиваем файлы на чанки по размеру
                chunks = self._create_chunks(zip_ref, file_list, chunk_size_mb)
                results["chunks"] = len(chunks)
                
                # Обрабатываем каждый чанк
                for i, chunk_files in enumerate(chunks):
                    print(f"Обработка чанка {i+1}/{len(chunks)} ({len(chunk_files)} файлов)")
                    
                    for file_name in chunk_files:
                        try:
                            # Создаем директорию если нужно
                            file_path = os.path.join(extract_to, file_name)
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)
                            
                            # Извлекаем файл
                            zip_ref.extract(file_name, extract_to)
                            results["extracted_files"] += 1
                            
                            # Добавляем в список файлов
                            results["file_list"].append(file_name)
                            
                        except Exception as e:
                            error_msg = f"Ошибка извлечения {file_name}: {str(e)}"
                            results["errors"].append(error_msg)
                            print(error_msg)
                    
                    # Возвращаем промежуточный прогресс
                    yield {
                        "chunk": i + 1,
                        "total_chunks": len(chunks),
                        "extracted_in_chunk": len(chunk_files),
                        "total_extracted": results["extracted_files"],
                        "status": "in_progress"
                    }
            
            print(f"Распаковано файлов: {results['extracted_files']}")
            yield {
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Ошибка распаковки архива {archive_path}: {e}"
            results["errors"].append(error_msg)
            print(error_msg)
            yield {
                "status": "error",
                "error": error_msg
            }
    
    def _create_chunks(self, zip_ref: zipfile.ZipFile, file_list: List[str], chunk_size_mb: int) -> List[List[str]]:
        """Создает чанки файлов по заданному размеру"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for file_name in file_list:
            try:
                file_info = zip_ref.getinfo(file_name)
                file_size = file_info.file_size
                
                # Если размер текущего чанка превышает лимит, создаем новый чанк
                if current_size + file_size > chunk_size_mb * 1024 * 1024 and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(file_name)
                current_size += file_size
                
            except:
                # Если не удалось получить информацию о файле, добавляем в текущий чанк
                current_chunk.append(file_name)
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def process_json_files_chunked(self, json_extract_dir: str, chunk_size: int = 10) -> Dict:
        """Обрабатывает JSON файлы чанками"""
        results = {
            "total_files": 0,
            "processed_files": 0,
            "new_resources": 0,
            "updated_resources": 0,
            "chunks": 0,
            "errors": [],
            "new_resources_list": []
        }
        
        try:
            # Собираем все JSON файлы
            json_files = []
            for root, dirs, files in os.walk(json_extract_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        json_files.append(os.path.join(root, file))
            
            results["total_files"] = len(json_files)
            
            if not json_files:
                print(f"В директории {json_extract_dir} не найдено JSON файлов")
                return results
            
            # Загружаем существующие ресурсы один раз
            resources_dist = {"resources": []}
            if os.path.exists(self.resources_dist_path):
                with open(self.resources_dist_path, 'r', encoding='utf-8') as f:
                    resources_dist = json.load(f)
            
            existing_resources = resources_dist.get('resources', [])
            existing_ids = self._get_existing_ids(existing_resources)
            
            # Разбиваем на чанки
            chunks = [json_files[i:i + chunk_size] 
                     for i in range(0, len(json_files), chunk_size)]
            results["chunks"] = len(chunks)
            
            for chunk_idx, chunk_files in enumerate(chunks):
                print(f"Обработка чанка JSON {chunk_idx + 1}/{len(chunks)}")
                
                chunk_results = {
                    "processed": 0,
                    "new": 0,
                    "updated": 0,
                    "errors": []
                }
                
                # Обрабатываем файлы в чанке
                for json_path in chunk_files:
                    try:
                        new_resources, new_count, updated_count = self._process_single_json_file(
                            json_path, existing_resources, existing_ids
                        )
                        
                        chunk_results["processed"] += 1
                        chunk_results["new"] += new_count
                        chunk_results["updated"] += updated_count
                        
                        if new_resources:
                            results["new_resources_list"].extend(new_resources)
                            
                    except Exception as e:
                        error_msg = f"Ошибка обработки {json_path}: {str(e)}"
                        chunk_results["errors"].append(error_msg)
                        results["errors"].append(error_msg)
                
                # Обновляем общие результаты
                results["processed_files"] += chunk_results["processed"]
                results["new_resources"] += chunk_results["new"]
                results["updated_resources"] += chunk_results["updated"]
                
                # Сохраняем промежуточные результаты
                self._save_intermediate_results(resources_dist)
                
                # Возвращаем прогресс
                yield {
                    "chunk": chunk_idx + 1,
                    "total_chunks": len(chunks),
                    "chunk_results": chunk_results,
                    "total_results": {
                        "processed": results["processed_files"],
                        "new": results["new_resources"],
                        "updated": results["updated_resources"],
                        "total": results["total_files"]
                    },
                    "status": "in_progress"
                }
            
            # Финальное сохранение
            self._save_final_results(resources_dist)
            
            yield {
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Ошибка обработки JSON файлов: {str(e)}"
            results["errors"].append(error_msg)
            yield {
                "status": "error",
                "error": error_msg,
                "partial_results": results
            }
    
    def _get_existing_ids(self, existing_resources: List[Dict]) -> List[int]:
        """Получает список существующих ID"""
        existing_ids = []
        for r in existing_resources:
            identificator = r.get('identificator', {})
            if isinstance(identificator, dict):
                resource_id = identificator.get('id', '')
                if resource_id.startswith('MEDIA_featurePhoto'):
                    try:
                        id_num = int(resource_id.replace('MEDIA_featurePhoto', ''))
                        existing_ids.append(id_num)
                    except ValueError:
                        continue
        return existing_ids
    
    def _process_single_json_file(self, json_path: str, existing_resources: List[Dict], existing_ids: List[int]) -> Tuple[List[Dict], int, int]:
        """Обрабатывает один JSON файл"""
        new_resources = []
        new_count = 0
        updated_count = 0
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            feature_data = data.get('featurePhoto2', {})
            if not feature_data:
                return [], 0, 0
                
            info_type = self.determine_information_type(feature_data.get('name_photo', ''))
            parent = feature_data.get('parent', '')
            uri_parent = parent.replace(' ', '_')
            
            # Генерируем новый ID
            new_id = max(existing_ids) + 1 if existing_ids else 1
            
            # Создаем новый ресурс
            new_resource = self._create_resource_from_data(
                feature_data, info_type, parent, uri_parent, new_id
            )
            
            # Проверяем на дубликаты
            is_duplicate, duplicate_resource, duplicate_idx = self.find_duplicate_resource(
                new_resource, existing_resources
            )
            
            if is_duplicate:
                # Обновляем существующий ресурс
                old_id = duplicate_resource['identificator']['id']
                new_resource['identificator']['id'] = old_id
                existing_resources[duplicate_idx] = new_resource
                updated_count = 1
                print(f"Обновлен ресурс: {old_id}")
            else:
                # Добавляем новый ресурс
                existing_resources.append(new_resource)
                existing_ids.append(new_id)
                new_count = 1
                new_resources.append(new_resource)
                print(f"Добавлен новый ресурс: MEDIA_featurePhoto{new_id}")
            
            return new_resources, new_count, updated_count
            
        except Exception as e:
            print(f"Ошибка обработки {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0
    
    def _create_resource_from_data(self, feature_data: Dict, info_type: str, parent: str, 
                                   uri_parent: str, new_id: int) -> Dict:
        """Создает ресурс из данных"""
        name_photo = feature_data.get('name_photo', '')
        file_name = os.path.basename(name_photo).replace(' ', '_')
        
        # Извлекаем относительный путь от images/
        relative_path = ""
        if 'images/' in name_photo:
            relative_path = name_photo.split('images/', 1)[1]
            path_parts = relative_path.split('/')
            path_parts = [part.replace(' ', '_') for part in path_parts]
            relative_path = '/'.join(path_parts)
        else:
            relative_path = file_name
        
        location_data = feature_data.get('location', {})
        coordinates = location_data.get('coordinates', {})
        
        lat_decimal = self.convert_coordinates(coordinates.get('latitude'))
        lon_decimal = self.convert_coordinates(coordinates.get('longitude'))
        
        class_info = feature_data.get('classification_info', {})
        result_info = class_info.get('result', {})
        
        flowering_info = feature_data.get('flowering', {})
        fruits_info = feature_data.get('fruits_present', {})
        
        flower_and_fruit_info = {}
        if info_type == 'flora':
            flower_and_fruit_info = {
                "flowering": flowering_info.get('flora_detector', ''),
                "fruits_present": fruits_info.get('flora_detector', '')
            }
            flower_color = feature_data.get('flower_color', {}).get('flora_detector')
            if flower_color:
                flower_and_fruit_info["flower_color"] = flower_color
        
        # Создаем ресурс
        resource = {
            "type": "Изображение",
            "identificator": {
                "id": f"MEDIA_featurePhoto{new_id}",
                "uri": f"istu.edu/va/baikal/daniil/{uri_parent}",
                "name": {
                    "common": result_info.get('name', parent),
                    "en_name": None,
                    "scientific": None,
                    "source": "Национальный парк/Заповедник"
                }
            },
            "access_options": {
                "author": feature_data.get('author_photo', ''),
                "file_path": f"https://testecobot.ru/images/{relative_path}",
                "source_url": "",
                "original_title": f"{file_name}. Фото {feature_data.get('author_photo', '')}",
                "rights": feature_data.get('rights', '')
            },
            "featurePhoto": {
                "name_photo": file_name,
                "parent": parent,
                "author_photo": feature_data.get('author_photo', ''),
                "name_object": result_info.get('name', parent),
                "season": feature_data.get('season', {}).get('result', ''),
                "sex": feature_data.get('sex', {}).get('result', ''),
                "habitat": feature_data.get('habitat', {}).get('result', ''),
                "flora_type": feature_data.get('class_type', {}).get('flora_type', {}).get('result', '') if info_type == 'flora' else '',
                "fauna_type": feature_data.get('class_type', {}).get('fauna_type', {}).get('result', '') if info_type == 'fauna' else '',
                "cloudiness": feature_data.get('cloudiness', {}).get('result', ''),
                "classification_info": {
                    "family": result_info.get('family', ''),
                    "genus": result_info.get('genus', ''),
                    "species": result_info.get('name', '')
                },
                "date": self.parse_date(feature_data.get('date_shooting_time', '')),
                "location": {
                    "country": "",
                    "region": "",
                    "coordinates": {
                        "latitude": lat_decimal,
                        "longitude": lon_decimal
                    }
                },
                "image_caption": feature_data.get('image_caption', {}).get('blip', ''),
                "yolo_detected_objects": feature_data.get('yolo_detected_objects', []),
                "flower_and_fruit_info": flower_and_fruit_info
            }
        }
        
        # Добавляем дополнительные поля
        feature_photo = resource['featurePhoto']
        for key in ['behavior', 'surface_type', 'placed', 'interaction', 'mood', 'age', 
                    'precipitation', 'temperature', 'wind', 'lifeform']:
            if key in feature_data:
                result_value = feature_data[key].get('result', '')
                if result_value and result_value not in ['Неопределено', 'Неопределён', '']:
                    feature_photo[key] = result_value
        
        return resource
    
    def _save_intermediate_results(self, resources_dist: Dict):
        """Сохраняет промежуточные результаты"""
        try:
            # Создаем временную копию для промежуточного сохранения
            temp_path = f"{self.resources_dist_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(resources_dist, f, ensure_ascii=False, indent=2)
            
            # Переименовываем обратно в основной файл
            os.replace(temp_path, self.resources_dist_path)
            
        except Exception as e:
            print(f"Ошибка промежуточного сохранения: {e}")
    
    def _save_final_results(self, resources_dist: Dict):
        """Сохраняет финальные результаты"""
        try:
            with open(self.resources_dist_path, 'w', encoding='utf-8') as f:
                json.dump(resources_dist, f, ensure_ascii=False, indent=2)
            print(f"Файл успешно сохранен: {self.resources_dist_path}")
            print(f"Всего ресурсов: {len(resources_dist.get('resources', []))}")
        except Exception as e:
            print(f"Ошибка финального сохранения: {e}")
    
    def process_images_chunked(self, images_extract_dir: str, chunk_size: int = 50) -> Dict:
        """Копирует изображения чанками"""
        results = {
            "total_files": 0,
            "copied_files": 0,
            "skipped_files": 0,
            "errors": [],
            "chunks": 0
        }
        
        try:
            # Собираем все изображения
            image_files = []
            for root, dirs, files in os.walk(images_extract_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        src_path = os.path.join(root, file)
                        image_files.append({
                            "src": src_path,
                            "rel_path": os.path.relpath(root, images_extract_dir),
                            "file_name": file
                        })
            
            results["total_files"] = len(image_files)
            
            if not image_files:
                print(f"В директории {images_extract_dir} не найдено изображений")
                return results
            
            # Создаем папку images если не существует
            os.makedirs(self.images_dir, exist_ok=True)
            
            # Разбиваем на чанки
            chunks = [image_files[i:i + chunk_size] 
                     for i in range(0, len(image_files), chunk_size)]
            results["chunks"] = len(chunks)
            
            for chunk_idx, chunk_images in enumerate(chunks):
                print(f"Обработка чанка изображений {chunk_idx + 1}/{len(chunks)}")
                
                chunk_results = {
                    "copied": 0,
                    "skipped": 0,
                    "errors": []
                }
                
                for img_info in chunk_images:
                    try:
                        src_path = img_info["src"]
                        rel_path = img_info["rel_path"]
                        file_name = img_info["file_name"]
                        
                        # Если путь начинается с 'images/', убираем эту часть
                        if rel_path.startswith('images/'):
                            rel_path = rel_path[7:]
                        elif rel_path == 'images':
                            rel_path = ''
                        
                        # Заменяем пробелы на подчеркивания в пути
                        if rel_path:
                            rel_parts = rel_path.split('/')
                            rel_parts = [part.replace(' ', '_') for part in rel_parts]
                            rel_path = '/'.join(rel_parts)
                        
                        # Определяем целевую папку
                        target_dir = os.path.join(self.images_dir, rel_path)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Заменяем пробелы на подчеркивания в имени файла
                        file_name = file_name.replace(' ', '_')
                        target_path = os.path.join(target_dir, file_name)
                        
                        # Копируем файл (с перезаписью если существует)
                        shutil.copy2(src_path, target_path)
                        chunk_results["copied"] += 1
                        
                    except Exception as e:
                        error_msg = f"Ошибка копирования {img_info['file_name']}: {str(e)}"
                        chunk_results["errors"].append(error_msg)
                        results["errors"].append(error_msg)
                        chunk_results["skipped"] += 1
                
                # Обновляем общие результаты
                results["copied_files"] += chunk_results["copied"]
                results["skipped_files"] += chunk_results["skipped"]
                
                # Возвращаем прогресс
                yield {
                    "chunk": chunk_idx + 1,
                    "total_chunks": len(chunks),
                    "chunk_results": chunk_results,
                    "total_results": {
                        "copied": results["copied_files"],
                        "skipped": results["skipped_files"],
                        "total": results["total_files"]
                    },
                    "status": "in_progress"
                }
            
            yield {
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Ошибка обработки изображений: {str(e)}"
            results["errors"].append(error_msg)
            yield {
                "status": "error",
                "error": error_msg,
                "partial_results": results
            }
    
    def process_upload_chunked(self, json_archive_path: Optional[str] = None, 
                           images_archive_path: Optional[str] = None,
                           reload_database: bool = False,
                           use_stubs: bool = True,
                           incremental: bool = True) -> Dict:
        """Основной метод обработки загрузки по чанкам"""
        results = {
            "json_processed": 0,
            "images_processed": 0,
            "new_resources": 0,
            "updated_resources": 0,
            "database_reloaded": False,
            "database_status": "not_started",
            "errors": [],
            "summary": {},
            "update_type": "без обновления БД"
        }
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Создаем временную папку
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Создана временная папка: {self.temp_dir}")
        
        new_resources_list = []
        json_extract_dir = None
        images_extract_dir = None
        
        try:
            # СНАЧАЛА ОБРАБАТЫВАЕМ ДАННЫЕ
            # Обработка архива с изображениями
            if images_archive_path and os.path.exists(images_archive_path):
                logger.info(f"Начинаем обработку архива изображений: {images_archive_path}")
                images_extract_dir = os.path.join(self.temp_dir, "images")
                os.makedirs(images_extract_dir, exist_ok=True)
                
                # Распаковываем архив
                for progress in self.extract_archive_chunked(images_archive_path, images_extract_dir):
                    yield {
                        "stage": "extract_images",
                        "progress": progress,
                        "message": "Распаковка архива с изображениями..."
                    }
                
                # Копируем изображения
                for progress in self.process_images_chunked(images_extract_dir):
                    yield {
                        "stage": "copy_images",
                        "progress": progress,
                        "message": "Копирование изображений..."
                    }
                
                # Обновляем результаты
                if progress.get("status") == "completed":
                    results["images_processed"] = progress["results"]["copied_files"]
            
            # Обработка архива с JSON аннотациями
            if json_archive_path and os.path.exists(json_archive_path):
                logger.info(f"Начинаем обработку JSON архива: {json_archive_path}")
                json_extract_dir = os.path.join(self.temp_dir, "json")
                os.makedirs(json_extract_dir, exist_ok=True)
                
                # Распаковываем архив
                for progress in self.extract_archive_chunked(json_archive_path, json_extract_dir):
                    yield {
                        "stage": "extract_json",
                        "progress": progress,
                        "message": "Распаковка архива с JSON..."
                    }
                
                # Обрабатываем JSON файлы
                for progress in self.process_json_files_chunked(json_extract_dir):
                    yield {
                        "stage": "process_json",
                        "progress": progress,
                        "message": "Обработка JSON файлов..."
                    }
                    
                    # Сохраняем результаты когда обработка завершится
                    if progress.get("status") == "completed":
                        results["json_processed"] = progress["results"]["processed_files"]
                        results["new_resources"] = progress["results"]["new_resources"]
                        results["updated_resources"] = progress["results"]["updated_resources"]
                        new_resources_list = progress["results"]["new_resources_list"]
            
            # ТОЛЬКО ПОСЛЕ ОБРАБОТКИ ДАННЫХ ПРОВЕРЯЕМ reload_database
            # Перезагрузка базы данных если запрошена
            if reload_database:
                logger.info("🚀 Запускаем перезагрузку базы данных...")
                
                # НЕ СОЗДАЕМ временный файл, postgres_adapter.py сам обрабатывает инкрементальный режим
                # Запускаем перезагрузку БД
                for progress in self.reload_database_chunked(
                    reload_database=reload_database,
                    use_stubs=use_stubs,
                    incremental=incremental,
                    new_resources_file=None  # Не передаем временный файл
                ):
                    yield {
                        "stage": "reload_database",
                        "progress": progress,
                        "message": "Перезагрузка базы данных..."
                    }
                
                # Обновляем статус базы данных
                if progress.get("status") == "completed":
                    results["database_reloaded"] = True
                    results["database_status"] = "completed"
                
                results["update_type"] = "полное" if not incremental else "инкрементальное"
            else:
                results["update_type"] = "без обновления БД"
            
            # Создаем резюме
            results["summary"] = self._create_summary(results)
            
            # Финальный результат
            yield {
                "stage": "completed",
                "results": results,
                "summary": results["summary"],
                "message": "Обработка завершена успешно!"
            }
            
        except Exception as e:
            error_msg = f"Ошибка в process_upload_chunked: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            
            yield {
                "stage": "error",
                "error": error_msg,
                "partial_results": results,
                "message": "Ошибка при обработке"
            }
            
        finally:
            # Удаляем временную папку
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Удалена временная папка: {self.temp_dir}")
                except Exception as e:
                    logger.error(f"Ошибка удаления временной папки: {e}")
                self.temp_dir = None
    
    def reload_database_chunked(self, reload_database: bool = False, 
                            use_stubs: bool = True,
                            incremental: bool = True,
                            new_resources_file: Optional[str] = None):
        """Перезагрузка базы данных с отслеживанием прогресса"""
        try:
            import sys
            import logging
            
            logger = logging.getLogger(__name__)
            
            logger.info("🔄 Начинаем перезагрузку базы данных...")
            
            # Путь к скриптам
            current_dir = Path(__file__).parent
            base_dir = current_dir.parent
            scripts_dir = base_dir / "knowledge_base_scripts" / "Relational"
            
            if not scripts_dir.exists():
                yield {
                    "status": "error",
                    "error": f"Директория не найдена: {scripts_dir}"
                }
                return
            
            # Запускаем postgres_adapter.py
            adapter_script = scripts_dir / "postgres_adapter.py"
            
            if adapter_script.exists():
                # Формируем команду БЕЗ --new-resources
                cmd = [sys.executable, str(adapter_script), "--json-file", str(self.resources_dist_path)]
                
                if use_stubs:
                    cmd.append("--use-stubs")
                
                if incremental:
                    cmd.append("--incremental")
                    logger.info("Режим: инкрементальное обновление")
                else:
                    cmd.append("--full")
                    logger.info("Режим: полная перезагрузка БД")
                
                # УБЕРИТЕ ЭТУ ЧАСТЬ - postgres_adapter.py не поддерживает --new-resources
                # if new_resources_file:
                #     cmd.extend(["--new-resources", str(new_resources_file)])
                
                logger.info(f"Команда: {' '.join(cmd)}")
                
                # Запускаем процесс
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=scripts_dir,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Читаем вывод в реальном времени
                while True:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    
                    if stdout_line:
                        stdout_line = stdout_line.rstrip()
                        if stdout_line:
                            # Определяем этап по содержанию сообщения
                            stage = self._determine_database_stage(stdout_line)
                            yield {
                                "status": "in_progress",
                                "stage": stage,
                                "message": stdout_line
                            }
                    
                    if stderr_line:
                        stderr_line = stderr_line.rstrip()
                        if stderr_line:
                            yield {
                                "status": "warning",
                                "message": stderr_line
                            }
                    
                    # Проверяем завершился ли процесс
                    if process.poll() is not None:
                        break
                
                # Ждем завершения
                returncode = process.wait(timeout=600)
                
                if returncode == 0:
                    yield {
                        "status": "completed",
                        "message": "База данных успешно обновлена"
                    }
                else:
                    yield {
                        "status": "error",
                        "error": f"Ошибка при обновлении БД (код: {returncode})"
                    }
                    
            else:
                yield {
                    "status": "error",
                    "error": "Скрипт postgres_adapter.py не найден"
                }
            
        except Exception as e:
            yield {
                "status": "error",
                "error": f"Ошибка перезагрузки базы данных: {str(e)}"
            }
    
    def _determine_database_stage(self, message: str) -> str:
        """Определяет этап обновления БД по сообщению"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['подключение', 'connection', 'connect']):
            return "connection"
        elif any(word in message_lower for word in ['создание', 'create', 'table']):
            return "tables"
        elif any(word in message_lower for word in ['вставка', 'insert', 'добавление']):
            return "insert"
        elif any(word in message_lower for word in ['индексы', 'index', 'индексирование']):
            return "indexing"
        elif any(word in message_lower for word in ['векторы', 'vectors', 'embeddings']):
            return "vectors"
        elif any(word in message_lower for word in ['завершено', 'completed', 'успешно']):
            return "completed"
        else:
            return "processing"
    
    def _create_summary(self, results: Dict) -> Dict:
        """Создает резюме обработки"""
        summary = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "operations": []
        }
        
        # Добавляем операции
        if results.get("json_processed", 0) > 0:
            summary["operations"].append({
                "name": "Обработка JSON файлов",
                "processed": results["json_processed"],
                "new_resources": results.get("new_resources", 0),
                "updated_resources": results.get("updated_resources", 0),
                "status": "completed"
            })
            summary["total_operations"] += 1
            summary["successful_operations"] += 1
        
        if results.get("images_processed", 0) > 0:
            summary["operations"].append({
                "name": "Копирование изображений",
                "processed": results["images_processed"],
                "status": "completed"
            })
            summary["total_operations"] += 1
            summary["successful_operations"] += 1
        
        if results.get("database_reloaded", False):
            summary["operations"].append({
                "name": "Обновление базы данных",
                "type": results.get("update_type", "неизвестно"),
                "status": "completed" if results.get("database_status") == "completed" else "started"
            })
            summary["total_operations"] += 1
            if results.get("database_status") == "completed":
                summary["successful_operations"] += 1
        
        if results.get("errors"):
            summary["failed_operations"] = len(results["errors"])
        
        return summary
    
    def reload_relational_database(self, reload_database: bool = False, 
                      use_stubs: bool = True,
                      incremental: bool = True,
                      new_resources_file: Optional[str] = None) -> bool:
        """Перезагружает или инкрементально обновляет реляционную базу данных"""
        try:
            import sys
            import logging
            
            logger = logging.getLogger(__name__)
            
            logger.info(f"🛠️  НАЧАЛО reload_relational_database - ВХОДНЫЕ ПАРАМЕТРЫ:")
            logger.info(f"🛠️  reload_database={reload_database}")
            logger.info(f"🛠️  incremental={incremental}")
            logger.info(f"🛠️  use_stubs={use_stubs}")
            logger.info(f"🛠️  new_resources_file={new_resources_file}")
            
            # Путь к скриптам
            current_dir = Path(__file__).parent  # core/
            base_dir = current_dir.parent  # родительская директория (где api.py)
            scripts_dir = base_dir / "knowledge_base_scripts" / "Relational"
            
            logger.info(f"📂 Ищем скрипты в: {scripts_dir}")
            
            if not scripts_dir.exists():
                logger.error(f"❌ Директория не найдена: {scripts_dir}")
                return False
            
            # Если нужно полное пересоздание БД
            if reload_database and not incremental:
                recreate_script = scripts_dir / "recreate_script.py"
                if recreate_script.exists():
                    logger.info("🔄 Запуск recreate_script.py для полной перезагрузки БД...")
                    result = subprocess.run(
                        [sys.executable, str(recreate_script)],
                        capture_output=True,
                        text=True,
                        cwd=scripts_dir,
                        timeout=10000
                    )
                    logger.info(f"recreate_script.py stdout: {result.stdout[:500]}...")
                    if result.stderr:
                        logger.error(f"recreate_script.py stderr: {result.stderr[:500]}...")
                    
                    if result.returncode != 0:
                        logger.error(f"❌ recreate_script.py завершился с ошибкой: {result.returncode}")
                        return False
            
            # Запускаем postgres_adapter.py
            adapter_script = scripts_dir / "postgres_adapter.py"
            
            if adapter_script.exists():
                logger.info(f"📄 Найден скрипт: {adapter_script}")
                
                # Если есть файл с новыми ресурсами, используем его
                json_file_to_use = self.resources_dist_path
                if new_resources_file and os.path.exists(new_resources_file):
                    json_file_to_use = new_resources_file
                    logger.info(f"📄 Используем файл только с новыми ресурсами: {new_resources_file}")
                
                # Формируем команду
                cmd = [sys.executable, str(adapter_script), "--json-file", str(json_file_to_use)]
                
                if use_stubs:
                    cmd.append("--use-stubs")
                
                # Определяем режим: полный или инкрементальный
                if incremental:
                    cmd.append("--incremental")
                    logger.info("🔧 Режим: инкрементальное обновление")
                else:
                    cmd.append("--full")
                    logger.info("🔧 Режим: полная перезагрузка БД")
                
                logger.info(f"🔧 Команда для запуска: {' '.join(cmd)}")
                
                try:
                    logger.info("🚀 Запускаем subprocess для postgres_adapter.py...")
                    
                    # Используем Popen для потоковой обработки вывода
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=scripts_dir,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Читаем вывод в реальном времени
                    while True:
                        stdout_line = process.stdout.readline()
                        stderr_line = process.stderr.readline()
                        
                        if stdout_line:
                            stdout_line = stdout_line.rstrip()
                            if stdout_line:  # Не логируем пустые строки
                                logger.info(f"📤 [postgres_adapter] {stdout_line}")
                        
                        if stderr_line:
                            stderr_line = stderr_line.rstrip()
                            if stderr_line:  # Не логируем пустые строки
                                logger.error(f"❌ [postgres_adapter] {stderr_line}")
                        
                        # Проверяем завершился ли процесс
                        if process.poll() is not None:
                            # Читаем остатки вывода
                            for stdout_line in process.stdout.readlines():
                                stdout_line = stdout_line.rstrip()
                                if stdout_line:
                                    logger.info(f"📤 [postgres_adapter] {stdout_line}")
                            
                            for stderr_line in process.stderr.readlines():
                                stderr_line = stderr_line.rstrip()
                                if stderr_line:
                                    logger.error(f"❌ [postgres_adapter] {stderr_line}")
                            break
                    
                    # Ждем завершения
                    returncode = process.wait(timeout=600)  # Увеличиваем таймаут до 10 минут
                    
                    logger.info(f"📊 Код возврата postgres_adapter.py: {returncode}")
                    
                    # Проверяем успешность выполнения
                    if returncode == 0:
                        logger.info("✅ База данных успешно обновлена")
                        return True
                    else:
                        logger.error(f"❌ Ошибка при обновлении БД (код возврата: {returncode})")
                        return False
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Таймаут выполнения postgres_adapter.py (больше 600 секунд)")
                    if process:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    return False
                except Exception as e:
                    logger.error(f"❌ Ошибка при запуске postgres_adapter.py: {e}")
                    return False
                    
            else:
                logger.error(f"❌ Скрипт postgres_adapter.py не найден")
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка перезагрузки базы данных: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
    def process_single_json_file(self, json_path: str) -> Tuple[List[Dict], int, int]:
        """Обрабатывает один JSON файл (упрощенная версия без чанков)"""
        new_resources = []
        new_count = 0
        updated_count = 0
        
        try:
            print(f"Начинаю обработку JSON файла: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Загружаем существующие ресурсы
            resources_dist = {"resources": []}
            if os.path.exists(self.resources_dist_path):
                with open(self.resources_dist_path, 'r', encoding='utf-8') as f:
                    resources_dist = json.load(f)
            
            print(f"Загружено существующих ресурсов: {len(resources_dist.get('resources', []))}")
            
            feature_data = data.get('featurePhoto2', {})
            if not feature_data:
                print(f"В файле {json_path} нет данных featurePhoto2")
                return [], 0, 0
                
            info_type = self.determine_information_type(feature_data.get('name_photo', ''))
            
            parent = feature_data.get('parent', '')
            uri_parent = parent.replace(' ', '_')
            
            # Генерируем новый ID если это новый ресурс
            existing_ids = []
            for r in resources_dist.get('resources', []):
                identificator = r.get('identificator', {})
                if isinstance(identificator, dict):
                    resource_id = identificator.get('id', '')
                    if resource_id.startswith('MEDIA_featurePhoto'):
                        try:
                            id_num = int(resource_id.replace('MEDIA_featurePhoto', ''))
                            existing_ids.append(id_num)
                        except ValueError:
                            continue
            
            new_id = max(existing_ids) + 1 if existing_ids else 1
            print(f"Новый ID для ресурса: MEDIA_featurePhoto{new_id}")
            
            name_photo = feature_data.get('name_photo', '')
            file_name = os.path.basename(name_photo).replace(' ', '_')
            
            # Извлекаем относительный путь от images/
            relative_path = ""
            if 'images/' in name_photo:
                # Берем часть после 'images/'
                relative_path = name_photo.split('images/', 1)[1]
                # Заменяем пробелы на подчеркивания в пути
                path_parts = relative_path.split('/')
                path_parts = [part.replace(' ', '_') for part in path_parts]
                relative_path = '/'.join(path_parts)
            else:
                # Если нет 'images/', используем имя файла
                relative_path = file_name
            
            print(f"Относительный путь изображения: {relative_path}")
            print(f"Имя файла: {file_name}")
            
            location_data = feature_data.get('location', {})
            coordinates = location_data.get('coordinates', {})
            
            lat_decimal = self.convert_coordinates(coordinates.get('latitude'))
            lon_decimal = self.convert_coordinates(coordinates.get('longitude'))
            
            class_info = feature_data.get('classification_info', {})
            result_info = class_info.get('result', {})
            
            flowering_info = feature_data.get('flowering', {})
            fruits_info = feature_data.get('fruits_present', {})
            
            flower_and_fruit_info = {}
            if info_type == 'flora':
                flower_and_fruit_info = {
                    "flowering": flowering_info.get('flora_detector', ''),
                    "fruits_present": fruits_info.get('flora_detector', '')
                }
                flower_color = feature_data.get('flower_color', {}).get('flora_detector')
                if flower_color:
                    flower_and_fruit_info["flower_color"] = flower_color
            
            # Создаем новый ресурс
            new_resource = {
                "type": "Изображение",
                "identificator": {
                    "id": f"MEDIA_featurePhoto{new_id}",
                    "uri": f"istu.edu/va/baikal/daniil/{uri_parent}",
                    "name": {
                        "common": result_info.get('name', parent),
                        "en_name": None,
                        "scientific": None,
                        "source": "Национальный парк/Заповедник"
                    }
                },
                "access_options": {
                    "author": feature_data.get('author_photo', ''),
                    "file_path": f"https://testecobot.ru/images/{relative_path}",
                    "source_url": "",
                    "original_title": f"{file_name}. Фото {feature_data.get('author_photo', '')}",
                    "rights": feature_data.get('rights', '')
                },
                "featurePhoto": {
                    "name_photo": file_name,
                    "parent": parent,
                    "author_photo": feature_data.get('author_photo', ''),
                    "name_object": result_info.get('name', parent),
                    "season": feature_data.get('season', {}).get('result', ''),
                    "sex": feature_data.get('sex', {}).get('result', ''),
                    "habitat": feature_data.get('habitat', {}).get('result', ''),
                    "flora_type": feature_data.get('class_type', {}).get('flora_type', {}).get('result', '') if info_type == 'flora' else '',
                    "fauna_type": feature_data.get('class_type', {}).get('fauna_type', {}).get('result', '') if info_type == 'fauna' else '',
                    "cloudiness": feature_data.get('cloudiness', {}).get('result', ''),
                    "classification_info": {
                        "family": result_info.get('family', ''),
                        "genus": result_info.get('genus', ''),
                        "species": result_info.get('name', '')
                    },
                    "date": self.parse_date(feature_data.get('date_shooting_time', '')),
                    "location": {
                        "country": "",
                        "region": "",
                        "coordinates": {
                            "latitude": lat_decimal,
                            "longitude": lon_decimal
                        }
                    },
                    "image_caption": feature_data.get('image_caption', {}).get('blip', ''),
                    "yolo_detected_objects": feature_data.get('yolo_detected_objects', []),
                    "flower_and_fruit_info": flower_and_fruit_info
                }
            }
            
            # Добавляем дополнительные поля
            feature_photo = new_resource['featurePhoto']
            
            for key in ['behavior', 'surface_type', 'placed', 'interaction', 'mood', 'age', 
                        'precipitation', 'temperature', 'wind', 'lifeform']:
                if key in feature_data:
                    result_value = feature_data[key].get('result', '')
                    if result_value and result_value not in ['Неопределено', 'Неопределён', '']:
                        feature_photo[key] = result_value
            
            print(f"Создан новый ресурс для: {file_name}")
            print(f"Путь к изображению: https://testecobot.ru/images/{relative_path}")
            
            # Проверяем на дубликаты
            is_duplicate, duplicate_resource, duplicate_idx = self.find_duplicate_resource(
                new_resource, resources_dist.get('resources', [])
            )
            
            print(f"Результат проверки дубликатов: is_duplicate={is_duplicate}, idx={duplicate_idx}")
            
            if is_duplicate:
                # Обновляем существующий ресурс
                old_id = duplicate_resource['identificator']['id']
                new_resource['identificator']['id'] = old_id
                resources_dist['resources'][duplicate_idx] = new_resource
                print(f"Обновлен существующий ресурс с ID: {old_id}")
                updated_count = 1
            else:
                # Добавляем новый ресурс
                resources_dist.setdefault('resources', []).append(new_resource)
                print(f"Добавлен новый ресурс с ID: MEDIA_featurePhoto{new_id}")
                new_count = 1
            
            # Сохраняем обновленный файл
            with open(self.resources_dist_path, 'w', encoding='utf-8') as f:
                json.dump(resources_dist, f, ensure_ascii=False, indent=2)
            
            print(f"Файл успешно сохранен: {self.resources_dist_path}")
            print(f"Всего ресурсов в файле: {len(resources_dist['resources'])}")
            
            new_resources.append(new_resource)
            return new_resources, new_count, updated_count
            
        except Exception as e:
            print(f"Ошибка обработки JSON файла {json_path}: {e}")
            import traceback
            traceback.print_exc()
            return [], 0, 0
        
    # Оригинальные методы для обратной совместимости
    def convert_coordinates(self, coord_str: str) -> Optional[float]:
        """Конвертирует координаты из строкового формата в десятичный"""
        if not coord_str:
            return None
        
        try:
            pattern = r'(\d+)°(\d+)\'([\d.]+)\"([NSEW])'
            match = re.match(pattern, coord_str)
            if match:
                degrees = float(match.group(1))
                minutes = float(match.group(2))
                seconds = float(match.group(3))
                direction = match.group(4)
                
                decimal = degrees + minutes/60 + seconds/3600
                
                if direction in ['S', 'W']:
                    decimal = -decimal
                
                return round(decimal, 6)
        except:
            pass
        
        return None
    
    def parse_date(self, date_str: str) -> str:
        """Парсит дату в стандартный формат"""
        if not date_str:
            return ""
        
        date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d.%m.%Y %H:%M:%S',
            '%d.%m.%Y',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d'
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        return date_str
    
    def determine_information_type(self, name_photo: str) -> str:
        """Определяет тип информации (flora/fauna)"""
        if 'flora' in name_photo.lower():
            return 'flora'
        elif 'fauna' in name_photo.lower():
            return 'fauna'
        return 'flora'
    
    def find_duplicate_resource(self, new_resource: Dict, existing_resources: List[Dict]) -> Tuple[bool, Optional[Dict], int]:
        """Ищет дубликат ресурса по полям access_options"""
        new_access = new_resource.get("access_options", {})
        
        for idx, resource in enumerate(existing_resources):
            if resource.get("type") != "Изображение":
                continue
                
            existing_access = resource.get("access_options", {})
            
            # Сравниваем все поля access_options кроме file_path (он может отличаться)
            fields_to_compare = ["author", "source_url", "original_title", "rights"]
            
            is_duplicate = all(
                new_access.get(field) == existing_access.get(field)
                for field in fields_to_compare
            )
            
            if is_duplicate:
                return True, resource, idx
                
        return False, None, -1