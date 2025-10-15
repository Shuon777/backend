from sentence_transformers import SentenceTransformer
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_config import embedding_config

def download_model(model_name: str):
    """Загрузить модель с HuggingFace"""
    model_path = embedding_config.MODEL_PATHS.get(model_name)
    
    if not model_path:
        raise ValueError(f"Модель {model_name} не настроена в конфигурации")
    
    if os.path.exists(model_path):
        if os.path.isfile(model_path):
            os.remove(model_path)
        else:
            shutil.rmtree(model_path)

    os.makedirs(model_path, exist_ok=True)

    print(f"Загрузка модели {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(model_path)
    print(f"Модель успешно сохранена в {model_path}")
    
    return model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Название модели для загрузки")
    args = parser.parse_args()
    
    download_model(args.model_name)