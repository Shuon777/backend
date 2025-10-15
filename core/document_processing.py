from langchain_core.documents import Document
import re
import json

def remove_links_from_doc(doc: Document) -> Document:
    """Создает копию документа без ссылок в контенте и метаданных"""
    if not doc.page_content:
        return doc
    
    uri_pattern = re.compile(
        r'(?:https?:\/\/|www\.|ftp:\/\/|file:\/\/)?'  
        r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'
        r'(?:\/[^\s]*)?'
        r'(?:\?[^\s]*)?'
        r'(?:\#[^\s]*)?'
    )
    
    cleaned_content = uri_pattern.sub('[URI_REMOVED]', doc.page_content)
    
    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        elif isinstance(d, str):
            return uri_pattern.sub('[URI_REMOVED]', d)
        return d
    
    cleaned_metadata = clean_dict(doc.metadata.copy()) if doc.metadata else {}
    
    return Document(
        page_content=cleaned_content,
        metadata=cleaned_metadata
    )
    
def find_resource_by_uri(file_path, uri):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, dict):
                data = data.get('resources', [])
            for resource in data:
                if isinstance(resource, dict) and resource.get('identificator', {}).get('uri') == uri:
                    return resource
    except Exception as e:
        print(f"Ошибка при поиске ресурса: {str(e)}")
    return None