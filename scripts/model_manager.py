import argparse
import sys
import os
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_embedding_model_from_HF import download_model
from embedding_config import embedding_config, get_model_dimension

def list_models():
    """Показать все доступные модели"""
    print("Доступные модели:")
    for model_name, model_path in embedding_config.MODEL_PATHS.items():
        current = " (текущая)" if model_name == embedding_config.current_model else ""
        print(f"  {model_name}{current} -> {model_path}")
# model_manager.py - исправленная функция run_script
def run_script(script_path, description, env_vars=None):
    """Запуск внешнего скрипта"""
    try:
        print(f"🚀 Запуск: {description}")
        print(f"📝 Скрипт: {script_path}")
        
        # Подготавливаем окружение
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=3600,
            env=env  # Передаем обновленное окружение
        )
        
        if result.returncode == 0:
            print(f"✅ {description} завершен успешно")
            if result.stdout:
                print(f"📋 Вывод:\n{result.stdout[:500]}...") 
            return True
        else:
            print(f"❌ Ошибка в {description}")
            print(f"🔴 Код возврата: {result.returncode}")
            if result.stderr:
                print(f"💥 Ошибка:\n{result.stderr}")
            if result.stdout:
                print(f"📋 Вывод:\n{result.stdout}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Таймаут при выполнении {description}")
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка при выполнении {description}: {e}")
        return False

# Обновленная функция rebuild_knowledge_base
def rebuild_knowledge_base():
    """Перестроение базы знаний после смены модели"""
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base_scripts", "Relational")
    
    # Получаем текущую модель и размерность
    current_model = embedding_config.current_model
    current_dimension = get_model_dimension(current_model)
    
    scripts_to_run = [
        {
            "path": os.path.join(scripts_dir, "recreate_script.py"),
            "description": "Пересоздание структуры базы данных",
            "env_vars": {
                "EMBEDDING_MODEL": current_model,
                "EMBEDDING_DIMENSION": str(current_dimension)
            }
        },
        {
            "path": os.path.join(scripts_dir, "postgres_adapter.py"),
            "description": "Импорт ресурсов в базу данных",
            "env_vars": {
                "EMBEDDING_MODEL": current_model,
                "EMBEDDING_DIMENSION": str(current_dimension)
            }
        },
        {
            "path": os.path.join(scripts_dir, "geojson_to_postgis.py"),
            "description": "Импорт геоданных в PostGIS",
            "env_vars": {
                "EMBEDDING_MODEL": current_model
            }
        }
    ]
    
    print("🔧 Начинается перестроение базы знаний...")
    print("=" * 60)
    
    success = True
    for script_info in scripts_to_run:
        if not run_script(script_info["path"], script_info["description"], script_info.get("env_vars")):
            success = False
            break
        print("-" * 40)
        time.sleep(2)
    
    if success:
        print("🎉 База знаний успешно перестроена!")
        return True
    else:
        print("💥 Перестроение базы знаний завершилось с ошибками")
        return False

def download_new_model(model_name: str, dimension: int = None):
    """Загрузить новую модель с указанием размерности"""
    try:
        print(f"📥 Загрузка модели: {model_name}")
        
        if dimension is None:
            dimension = get_model_dimension(model_name)
            print(f"ℹ️  Автоопределение размерности: {dimension}")
        
        path = download_model(model_name)
        print(f"✅ Модель {model_name} загружена в: {path}")
        
        if model_name not in embedding_config.MODEL_PATHS:
            new_path = os.path.join(embedding_config.BASE_MODELS_DIR, model_name.replace("/", "_"))
            embedding_config.MODEL_PATHS[model_name] = new_path
            
            if hasattr(embedding_config, 'MODEL_DIMENSIONS'):
                embedding_config.MODEL_DIMENSIONS[model_name] = dimension
            else:
                embedding_config.MODEL_DIMENSIONS = {model_name: dimension}
            
            print(f"➕ Модель {model_name} добавлена в конфигурацию (размерность: {dimension})")
            
        return path, dimension
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None, None

def set_active_model(model_name: str, rebuild_kb: bool = True):
    """Установить активную модель с опциональным перестроением БЗ"""
    try:
        if model_name == embedding_config.current_model:
            print(f"ℹ️ Модель {model_name} уже активна")
            return True
            
        print(f"🔄 Установка активной модели: {model_name}")
        
        new_dimension = get_model_dimension(model_name)
        current_dimension = get_model_dimension(embedding_config.current_model)
        
        embedding_config.set_active_model(model_name)
        
        os.environ["EMBEDDING_MODEL"] = model_name
        os.environ["EMBEDDING_DIMENSION"] = str(new_dimension)
        
        print(f"✅ Активная модель установлена: {model_name}")
        print(f"📁 Путь: {embedding_config.current_model_path}")
        print(f"📏 Размерность: {new_dimension}")
        
        dimension_changed = new_dimension != current_dimension
        
        if rebuild_kb or dimension_changed:
            if dimension_changed:
                print(f"⚠️  ВНИМАНИЕ: Размерность изменилась ({current_dimension} -> {new_dimension})")
                print("🔄 Требуется полное перестроение базы знаний")
            
            print("\n" + "=" * 60)
            print("🔄 Требуется перестроение базы знаний")
            print("=" * 60)
            
            confirm = input("Продолжить перестроение базы знаний? (y/N): ")
            if confirm.lower() in ['y', 'yes', 'д', 'да']:
                return rebuild_knowledge_base()
            else:
                print("❌ Перестроение базы знаний отменено")
                print("⚠️  ВАЖНО: База данных может быть несовместима с новой моделью!")
                return False
        else:
            print("ℹ️ Перестроение базы знаний не требуется")
            return True
            
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        list_models()
        return False
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Менеджер эмбеддинговых моделей")
    parser.add_argument("--list", action="store_true", help="Показать все модели")
    parser.add_argument("--set", type=str, help="Установить активную модель")
    parser.add_argument("--download", type=str, help="Загрузить новую модель")
    parser.add_argument("--dimension", type=int, help="Размерность эмбеддингов для новой модели")
    parser.add_argument("--no-rebuild", action="store_true", help="Не перестраивать БЗ после смены модели")
    parser.add_argument("--rebuild-only", action="store_true", help="Только перестроить БЗ без смены модели")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.set:
        success = set_active_model(args.set, not args.no_rebuild)
        if not success:
            sys.exit(1)
    elif args.download:
        download_new_model(args.download, args.dimension)
    elif args.rebuild_only:
        success = rebuild_knowledge_base()
        if not success:
            sys.exit(1)
    else:
        print("📊 Текущая активная модель:")
        model_name, model_path = embedding_config.get_active_model()
        dimension = get_model_dimension(model_name)
        print(f"  🎯 {model_name} -> {model_path}")
        print(f"  📏 Размерность: {dimension}")
        print(f"\n💡 Используйте --list чтобы увидеть все доступные модели")
        print(f"💡 Используйте --set <model> чтобы сменить модель")
        print(f"💡 Используйте --download <model> --dimension <size> чтобы загрузить новую модель")
        print(f"💡 Используйте --rebuild-only чтобы перестроить БЗ")