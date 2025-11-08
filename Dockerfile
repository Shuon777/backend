# /var/www/salut_bot/Dockerfile
FROM python:3.10

WORKDIR /app

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Копируем весь код проекта
COPY . .

# Команда запуска
CMD ["gunicorn", "--bind", "0.0.0.0:5555", "--workers", "2", "api:app"]