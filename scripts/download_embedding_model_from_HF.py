# download_embedding_model_from_HF.py
import argparse
import os
import shutil
import sys
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding_config import embedding_config

def download_model(model_name: str, model_type: str = 'embedding'):
    """Загрузить модель с HuggingFace"""
    if model_type == 'embedding':
        model_path = embedding_config.MODEL_PATHS.get(model_name)
        if not model_path:
            raise ValueError(f"Модель {model_name} не настроена в конфигурации")
        # Удаляем старую версию
        if os.path.exists(model_path):
            if os.path.isfile(model_path):
                os.remove(model_path)
            else:
                shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)

        print(f"Загрузка модели эмбеддингов {model_name}...")
        model = SentenceTransformer(model_name)
        model.save(model_path)
        print(f"Модель успешно сохранена в {model_path}")

    elif model_type == 'reranker':
        # Для реранкера используем отдельную папку
        base_dir = os.path.dirname(embedding_config.BASE_MODELS_DIR)  # на уровень выше?
        # Лучше создать подпапку rerankers внутри embedding_models
        reranker_dir = os.path.join(embedding_config.BASE_MODELS_DIR, "rerankers")
        os.makedirs(reranker_dir, exist_ok=True)

        # Формируем имя папки из названия модели (заменяем '/' на '_')
        safe_name = model_name.replace('/', '_')
        model_path = os.path.join(reranker_dir, safe_name)

        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path, exist_ok=True)

        print(f"Загрузка модели реранкера {model_name}...")
        model = CrossEncoder(model_name)
        model.save(model_path)
        print(f"Модель реранкера успешно сохранена в {model_path}")

    else:
        raise ValueError("model_type должен быть 'embedding' или 'reranker'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Название модели для загрузки")
    parser.add_argument("--type", choices=['embedding', 'reranker'], default='embedding',
                        help="Тип модели: embedding или reranker")
    args = parser.parse_args()
    download_model(args.model_name, args.type)