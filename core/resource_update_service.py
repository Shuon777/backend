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


class ResourceUpdateService:
    def __init__(self, resources_dist_path: str, images_dir: str):
        self.resources_dist_path = resources_dist_path
        self.images_dir = images_dir
        self.temp_dir = None
        
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """Распаковывает архив во временную папку"""
        try:
            print(f"Распаковка архива {archive_path} в {extract_to}")
            
            # Создаем папку для распаковки если не существует
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Получаем список файлов в архиве
                file_list = zip_ref.namelist()
                print(f"Файлов в архиве: {len(file_list)}")
                
                # Распаковываем все файлы
                zip_ref.extractall(extract_to)
                
                # Логируем структуру распакованных файлов
                extracted_files = []
                for root, dirs, files in os.walk(extract_to):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), extract_to)
                        extracted_files.append(rel_path)
                
                print(f"Распаковано файлов: {len(extracted_files)}")
                if extracted_files:
                    print(f"Первые 10 файлов: {extracted_files[:10]}")
                
            return True
        except Exception as e:
            print(f"Ошибка распаковки архива {archive_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
    def find_duplicate_resource(self, new_resource: Dict, existing_resources: List[Dict]) -> Tuple[bool, Optional[Dict]]:
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
    
    def process_json_file(self, json_path: str):
        """Обрабатывает один JSON файл с аннотациями
        Возвращает: (список новых ресурсов, количество новых, количество обновленных)
        """
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
    
    def process_images(self, images_extract_dir: str):
        """Копирует изображения из временной папки в основную папку images"""
        try:
            # Создаем папку images если не существует
            os.makedirs(self.images_dir, exist_ok=True)
            
            # Рекурсивно копируем все изображения
            for root, dirs, files in os.walk(images_extract_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        src_path = os.path.join(root, file)
                        
                        # Определяем относительный путь от корня извлечения
                        rel_path = os.path.relpath(root, images_extract_dir)
                        
                        # Если путь начинается с 'images/', убираем эту часть
                        if rel_path.startswith('images/'):
                            rel_path = rel_path[7:]  # Убираем 'images/'
                        elif rel_path == 'images':
                            rel_path = ''
                        
                        # Заменяем пробелы на подчеркивания в пути
                        if rel_path:
                            rel_parts = rel_path.split('/')
                            rel_parts = [part.replace(' ', '_') for part in rel_parts]
                            rel_path = '/'.join(rel_parts)
                        
                        # Определяем целевую папку
                        target_dir = os.path.join(self.images_dir, rel_path)
                        
                        # Создаем целевую папку если нужно
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Заменяем пробелы на подчеркивания в имени файла
                        file_name = file.replace(' ', '_')
                        
                        # Копируем файл (с перезаписью если существует)
                        target_path = os.path.join(target_dir, file_name)
                        shutil.copy2(src_path, target_path)
                        print(f"Скопировано изображение: {target_path}")
            
        except Exception as e:
            print(f"Ошибка копирования изображений: {e}")
            import traceback
            traceback.print_exc()
    
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
                        timeout=300
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
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=scripts_dir,
                        timeout=300  # 5 минут таймаут
                    )
                    
                    logger.info(f"📤 postgres_adapter.py stdout (первые 500 символов):")
                    logger.info(result.stdout[:500])
                    
                    if result.stdout and len(result.stdout) > 500:
                        logger.info(f"... (еще {len(result.stdout) - 500} символов)")
                    
                    if result.stderr:
                        logger.error(f"❌ postgres_adapter.py stderr:")
                        logger.error(result.stderr[:500])
                        if len(result.stderr) > 500:
                            logger.error(f"... (еще {len(result.stderr) - 500} символов)")
                    
                    logger.info(f"📊 Код возврата: {result.returncode}")
                    
                    # Проверяем успешность выполнения
                    if result.returncode == 0:
                        logger.info("✅ База данных успешно обновлена")
                        return True
                    else:
                        logger.error(f"❌ Ошибка при обновлении БД (код возврата: {result.returncode})")
                        return False
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"❌ Таймаут выполнения postgres_adapter.py (больше 300 секунд)")
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
    
    def reload_database_only(self, reload_database: bool = False, use_stubs: bool = True, incremental: bool = True) -> Dict:
        """Метод только для перезагрузки базы данных без обработки архивов"""
        results = {
            "json_processed": 0,
            "images_processed": 0,
            "new_resources": 0,
            "updated_resources": 0,
            "database_reloaded": False,
            "errors": []
        }
        
        try:
            # Перезагрузка базы данных
            print(f"🚀 Вызываем reload_relational_database для перезагрузки БД")
            results["database_reloaded"] = self.reload_relational_database(
                reload_database=reload_database,
                use_stubs=use_stubs,
                incremental=incremental
            )
            
            if results["database_reloaded"]:
                print("✅ База данных успешно перезагружена")
                results["update_type"] = "полное" if not incremental else "инкрементальное"
            else:
                results["errors"].append("Не удалось перезагрузить базу данных")
                print("❌ Ошибка при перезагрузке базы данных")
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            results["errors"].append(error_msg)
            print(f"Ошибка в reload_database_only: {error_msg}")
            import traceback
            traceback.print_exc()
            return results

    def process_upload(self, json_archive_path: Optional[str] = None, 
                images_archive_path: Optional[str] = None,
                reload_database: bool = False,
                use_stubs: bool = True,
                incremental: bool = True) -> Dict:
        """Основной метод обработки загрузки"""
        results = {
            "json_processed": 0,
            "images_processed": 0,
            "new_resources": 0,
            "updated_resources": 0,
            "database_reloaded": False,
            "errors": [],
            "update_type": "без обновления БД"
        }
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Создаем временную папку
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Создана временная папка: {self.temp_dir}")
        
        try:
            # Проверяем, есть ли что обрабатывать
            has_archives = (json_archive_path and os.path.exists(json_archive_path)) or \
                        (images_archive_path and os.path.exists(images_archive_path))
            
            if not has_archives and not reload_database:
                results["errors"].append("Не указаны архивы для обработки и не запрошена перезагрузка БД")
                return results
            
            # Обработка архивов...
            new_resources_list = []  # Будем собирать новые ресурсы
            
            # Обработка архива с изображениями
            if images_archive_path and os.path.exists(images_archive_path):
                logger.info(f"Обработка архива изображений: {images_archive_path}")
                images_extract_dir = os.path.join(self.temp_dir, "images")
                os.makedirs(images_extract_dir, exist_ok=True)
                
                if self.extract_archive(images_archive_path, images_extract_dir):
                    logger.info(f"Архив распакован в: {images_extract_dir}")
                    self.process_images(images_extract_dir)
                    # Подсчитываем количество обработанных изображений
                    image_count = 0
                    for root, dirs, files in os.walk(images_extract_dir):
                        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                        image_count += len(image_files)
                    results["images_processed"] = image_count
                    logger.info(f"Обработано изображений: {image_count}")
            
            # Обработка архива с JSON аннотациями
            if json_archive_path and os.path.exists(json_archive_path):
                logger.info(f"Обработка JSON архива: {json_archive_path}")
                json_extract_dir = os.path.join(self.temp_dir, "json")
                os.makedirs(json_extract_dir, exist_ok=True)
                
                if self.extract_archive(json_archive_path, json_extract_dir):
                    logger.info(f"JSON архив распакован в: {json_extract_dir}")
                    # Рекурсивно ищем JSON файлы
                    json_count = 0
                    new_total = 0
                    updated_total = 0
                    
                    for root, dirs, files in os.walk(json_extract_dir):
                        for file in files:
                            if file.lower().endswith('.json'):
                                json_path = os.path.join(root, file)
                                logger.info(f"Обработка JSON файла: {json_path}")
                                processed, new_count, updated_count = self.process_json_file(json_path)
                                json_count += 1
                                new_total += new_count
                                updated_total += updated_count
                                
                                # Сохраняем новые ресурсы
                                if processed:
                                    new_resources_list.extend(processed)
                    
                    results["json_processed"] = json_count
                    results["new_resources"] = new_total
                    results["updated_resources"] = updated_total
                    logger.info(f"Обработано JSON файлов: {json_count}")
                    logger.info(f"Добавлено новых ресурсов: {new_total}")
                    logger.info(f"Обновлено существующих ресурсов: {updated_total}")
            
            # Перезагрузка базы данных если запрошена
            logger.info(f"ПРОВЕРКА: reload_database = {reload_database}")
            if reload_database:
                logger.info(f"🚀 ЗАПРОШЕНА перезагрузка БД, вызываем reload_relational_database...")
                
                # Если есть новые ресурсы и это инкрементальное обновление,
                # создаем временный файл только с новыми ресурсами
                temp_json_file = None
                if incremental and new_resources_list and len(new_resources_list) > 0:
                    temp_json_file = os.path.join(self.temp_dir, "new_resources.json")
                    with open(temp_json_file, 'w', encoding='utf-8') as f:
                        json.dump({"resources": new_resources_list}, f, ensure_ascii=False, indent=2)
                    logger.info(f"📄 Создан временный файл с {len(new_resources_list)} новыми ресурсами: {temp_json_file}")
                
                results["database_reloaded"] = self.reload_relational_database(
                    reload_database=reload_database,
                    use_stubs=use_stubs,
                    incremental=incremental,
                    new_resources_file=temp_json_file if temp_json_file else None
                )
                
                logger.info(f"Результат reload_relational_database: {results['database_reloaded']}")
                
                if results["database_reloaded"]:
                    results["update_type"] = "полное" if not incremental else "инкрементальное"
                    logger.info(f"✅ База данных успешно обновлена ({results['update_type']})")
                else:
                    logger.error(f"❌ Ошибка при обновлении базы данных")
                    results["errors"].append("Не удалось обновить базу данных")
            else:
                logger.info("⏭️  Обновление базы данных не запрошено")
                results["update_type"] = "без обновления БД"
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            results["errors"].append(error_msg)
            logger.error(f"Ошибка в process_upload: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            return results
        finally:
            # Удаляем временную папку
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Удалена временная папка: {self.temp_dir}")
                except Exception as e:
                    logger.error(f"Ошибка удаления временной папки: {e}")
                self.temp_dir = None