import os

def list_files(startpath, output_file):
    excluded_dirs = {'node_modules', '.next', '.git','public','__pycache__','pgvector'}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            # Удаляем исключенные директории из списка
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            # Печатаем путь к текущей директории
            f.write(root + '\n')
            
            # Печатаем файлы в текущей директории
            for file in files:
                f.write(f'    {file}\n')

# Используем текущую директорию
project_path = os.getcwd()
output_file = 'project_structure.txt'
list_files(project_path, output_file)

print(f'Структура проекта сохранена в {output_file}')
