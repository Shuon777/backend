# use_cases/index_resources.py
import logging
import sys
from typing import List

from ..domain.interfaces import (
    ResourceProvider, TextExtractor, Chunker,
    EmbeddingService, VectorStoreService
)


class IndexResourcesUseCase:
    def __init__(
        self,
        provider: ResourceProvider,
        extractor: TextExtractor,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ):
        self._provider = provider
        self._extractor = extractor
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._logger = logging.getLogger(__name__)

    def execute(self, output_path: str) -> None:
        resources = self._provider.get_resources()
        all_chunks = []
        total_resources = len(resources)
        
        print(f"\n=== НАЧАЛО ИНДЕКСАЦИИ ===")
        print(f"Всего ресурсов: {total_resources}")
        print(f"Обработка ресурсов...\n")
        
        processed = 0
        skipped_no_text = 0
        skipped_image_desc = 0
        
        for idx, res in enumerate(resources, 1):
            try:
                text = self._extractor.extract(res)
                
                if not text:
                    if res.title and res.title.startswith("Описание изображения:"):
                        skipped_image_desc += 1
                    else:
                        skipped_no_text += 1
                    continue
                
                metadata = {
                    "resource_id": res.resource_id,
                    "title": res.title or "Без названия",
                    "modality_type": res.modality_type,
                }
                chunks = self._chunker.chunk(text, metadata)
                all_chunks.extend(chunks)
                processed += 1
                
                print(f"[{idx}/{total_resources}] ✓ Обработан: {res.resource_id} -> {len(chunks)} чанков")
                
            except Exception as e:
                self._logger.error(f"Failed to process resource {res.resource_id}: {e}")
                print(f"[{idx}/{total_resources}] ✗ Ошибка: {res.resource_id} - {e}")
        
        print(f"\n=== СТАТИСТИКА ===")
        print(f"Обработано успешно: {processed}")
        print(f"Пропущено (нет текста): {skipped_no_text}")
        print(f"Пропущено (описания изображений): {skipped_image_desc}")
        print(f"Всего чанков создано: {len(all_chunks)}")
        
        if not all_chunks:
            raise RuntimeError("No chunks generated")
        
        print(f"\nГенерация эмбеддингов...")
        texts = [chunk.text for chunk in all_chunks]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"  Эмбеддинги: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")
            embeddings = self._embedding_service.embed(batch)
            all_embeddings.extend(embeddings)
        
        print(f"\nСохранение в векторную БД...")
        self._vector_store.add_documents(all_chunks, all_embeddings)
        self._vector_store.save(output_path)
        
        print(f"\n=== ГОТОВО ===")
        print(f"Индекс сохранен: {output_path}")
        print(f"Всего чанков: {len(all_chunks)}")
        print(f"Размерность эмбеддингов: {len(all_embeddings[0]) if all_embeddings else 0}")
        
        self._logger.info(f"Index saved to {output_path}, chunks={len(all_chunks)}")