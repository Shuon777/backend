from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorIndexingConfig:
    resources_json_path: Path
    output_index_dir: Path
    embedding_model_path: str
    chunk_size: int = 512
    device: str = 'cpu'