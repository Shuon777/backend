FROM python:3.10
WORKDIR /app

COPY requirements.txt .
# Устанавливаем gunicorn вместе со всеми зависимостями
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Увеличим количество воркеров для лучшей производительности
# Формула: (2 * кол-во ядер CPU) + 1
CMD ["python", "api.py"]