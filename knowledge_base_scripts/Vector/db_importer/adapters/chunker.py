# adapters/chunker.py
import re
from typing import List, Dict, Any

from ..domain.entities import TextChunk
from ..domain.interfaces import Chunker


class FixedSizeChunker(Chunker):
    def __init__(self, max_chunk_size: int = 512, overlap_size: int = 50):
        self._max_size = max_chunk_size
        self._overlap = overlap_size

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        if not text:
            return []
        if len(text) <= self._max_size:
            return [TextChunk(text=text, metadata=metadata)]

        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for para in paragraphs:
            if len(para) <= self._max_size:
                chunks.append(TextChunk(text=para, metadata=metadata))
            else:
                chunks.extend(self._split_long_text(para, metadata))
        return chunks

    def _split_long_text(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current = []
        current_len = 0
        result = []

        for sent in sentences:
            sent_len = len(sent)
            if sent_len > self._max_size:
                if current:
                    result.append(TextChunk(text=' '.join(current), metadata=metadata))
                sub_chunks = self._split_oversized_sentence(sent)
                for sub in sub_chunks:
                    result.append(TextChunk(text=sub, metadata=metadata))
                current = []
                current_len = 0
                continue

            if current_len + sent_len + 2 > self._max_size:
                if current:
                    result.append(TextChunk(text=' '.join(current), metadata=metadata))
                current = [sent]
                current_len = sent_len
            else:
                current.append(sent)
                current_len += sent_len + 2
        if current:
            result.append(TextChunk(text=' '.join(current), metadata=metadata))
        return result

    def _split_oversized_sentence(self, sentence: str) -> List[str]:
        chunks = []
        start = 0
        length = len(sentence)
        while start < length:
            end = min(start + self._max_size, length)
            if end < length:
                while end > start and sentence[end] not in ' .,;:!?':
                    end -= 1
                if end == start:
                    end = min(start + self._max_size, length)
            chunks.append(sentence[start:end].strip())
            start = end - self._overlap if end < length else length
        return chunks