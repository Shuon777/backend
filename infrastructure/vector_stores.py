import logging
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def init_vector_stores(faiss_index_path, embedding_model_path):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    
    vectorstores = {}
    store_types = ["Текст", "Изображение", "Картографическая информация"]
    
    for store_type in store_types:
        try:
            store_path = os.path.join(faiss_index_path, store_type)
            if os.path.exists(store_path):
                vectorstores[store_type] = FAISS.load_local(
                    store_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Успешно загружено хранилище для типа: {store_type}")
            else:
                logger.warning(f"Путь к хранилищу не существует: {store_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки хранилища {store_type}: {str(e)}")
    
    return vectorstores