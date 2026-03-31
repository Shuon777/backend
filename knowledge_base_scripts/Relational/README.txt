db_importer.main - новая информационная модель - разрабатывается
Относительные импорты не работают при прямом запуске (python script.py), требуется запуск как модуля (python -m package.module): python -m db_importer.main --full

recreate_script.py + postgres_adapter.py - рабочая версия старой модели, которая сейчас используется