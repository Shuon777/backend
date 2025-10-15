from sentence_transformers import SentenceTransformer
import os
import shutil

model_path = '/var/www/salut_bot/embedding_model'

if os.path.exists(model_path):
    if os.path.isfile(model_path):
        os.remove(model_path)
    else:
        shutil.rmtree(model_path)

os.makedirs(model_path, exist_ok=True)

print("Загрузка модели...")
model = SentenceTransformer('sergeyzh/BERTA')
model.save(model_path)
print(f"Модель успешно сохранена в {model_path}")

print("\nСодержимое директории:")
print(os.listdir(model_path))