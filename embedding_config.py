import os
import json
from pathlib import Path

MODEL_DIMENSIONS = {
    "sergeyzh/BERTA": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384
}   

def get_model_dimension(model_name):
    """Получить размерность для модели"""
    return MODEL_DIMENSIONS.get(model_name, 768)

class EmbeddingConfig:
    def __init__(self):
        # ИСПРАВЛЕНО: используем абсолютный путь
        current_dir = Path(__file__).parent
        self.BASE_MODELS_DIR = str(current_dir.parent / "embedding_models")  # Поднимаемся на уровень выше
        
        self.CONFIG_FILE = os.path.join(self.BASE_MODELS_DIR, "active_model.json")
        
        os.makedirs(self.BASE_MODELS_DIR, exist_ok=True)
        
        # Исправлено: используем полные имена моделей
        self.DEFAULT_MODEL = "sergeyzh/BERTA"
        self.ALTERNATIVE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        
        self.MODEL_PATHS = {
            "sergeyzh/BERTA": os.path.join(self.BASE_MODELS_DIR, "BERTA"),
            "sentence-transformers/all-MiniLM-L6-v2": os.path.join(self.BASE_MODELS_DIR, "all-MiniLM-L6-v2")
        }
        
        self.current_model = self._load_active_model()
        self.current_model_path = self.get_model_path(self.current_model)
        
        # Добавим отладочную информацию
        print(f"📁 Базовая директория моделей: {self.BASE_MODELS_DIR}")
        print(f"🎯 Активная модель: {self.current_model}")
        print(f"📁 Путь к модели: {self.current_model_path}")
        print(f"📏 Размерность: {get_model_dimension(self.current_model)}")
    
    def _load_active_model(self):
        """Загружает активную модель из файла конфигурации"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # Обработка старого формата (просто "BERTA")
                    active_model = config.get('active_model', self.DEFAULT_MODEL)
                    if active_model == "BERTA":
                        return "sergeyzh/BERTA"
                    return active_model
        except:
            pass
        
        # Обработка переменной окружения
        env_model = os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)
        if env_model == "BERTA":
            return "sergeyzh/BERTA"
        return env_model
    
    def _save_active_model(self):
        """Сохраняет активную модель в файл конфигурации"""
        try:
            config = {
                'active_model': self.current_model,
                'model_path': self.current_model_path,
                'dimension': get_model_dimension(self.current_model)
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
    
    def get_model_path(self, model_name: str) -> str:
        """Получить путь к модели по имени"""
        # Обработка короткого имени "BERTA"
        if model_name == "BERTA":
            model_name = "sergeyzh/BERTA"
        return self.MODEL_PATHS.get(model_name, self.MODEL_PATHS[self.DEFAULT_MODEL])
    
    def set_active_model(self, model_name: str):
        """Установить активную модель"""
        # Обработка короткого имени "BERTA"
        if model_name == "BERTA":
            model_name = "sergeyzh/BERTA"
            
        if model_name in self.MODEL_PATHS:
            self.current_model = model_name
            self.current_model_path = self.MODEL_PATHS[model_name]
            self._save_active_model()  # Сохраняем в файл
        else:
            raise ValueError(f"Модель {model_name} не найдена в конфигурации")
    
    def get_active_model(self) -> tuple:
        """Получить текущую активную модель и путь"""
        return self.current_model, self.current_model_path

embedding_config = EmbeddingConfig()