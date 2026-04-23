#!/usr/bin/env python3
import sys
import json
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

from .infrastructure.vector_indexing_config import VectorIndexingConfig
from .adapters import (
    JsonResourceProvider,
    NewResourceTextExtractor,
    FixedSizeChunker,
    HuggingFaceEmbeddingService,
    FaissVectorStore
)
from .use_cases import IndexResourcesUseCase


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def get_active_model_path() -> str:
    base_dir = Path("/var/www/salut_bot")
    config_path = base_dir / "embedding_models" / "active_model.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"active_model.json not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    model_path = config.get('model_path')
    if not model_path:
        raise ValueError("model_path not found in active_model.json")
    
    # Поддержка относительных путей
    path_obj = Path(model_path)
    if not path_obj.is_absolute():
        path_obj = base_dir / model_path
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Model directory not found: {path_obj}")
    
    return str(path_obj)


def parse_args() -> tuple:
    import argparse
    parser = argparse.ArgumentParser(description='Build vector index from resources.json')
    parser.add_argument(
        '--resources-file',
        type=str,
        default='/var/www/salut_bot/json_files/resources.json',
        help='Path to resources.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/var/www/salut_bot/knowledge_base_scripts/Vector/new_faiss_index',
        help='Output directory for FAISS index'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to local embedding model (overrides active_model.json)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=512,
        help='Maximum chunk size in characters'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for embedding model'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    resources_path = Path(args.resources_file)
    if not resources_path.exists():
        print(f"Error: Resources file not found: {resources_path}", file=sys.stderr)
        sys.exit(1)

    if args.model_path:
        model_path = args.model_path
    else:
        try:
            model_path = get_active_model_path()
            print(f"Using active model: {model_path}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Please specify --model-path or ensure active_model.json exists", file=sys.stderr)
            sys.exit(1)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"Error: Embedding model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    config = VectorIndexingConfig(
        resources_json_path=resources_path,
        output_index_dir=Path(args.output_dir),
        embedding_model_path=model_path,
        chunk_size=args.chunk_size,
        device=args.device
    )

    provider = JsonResourceProvider(config.resources_json_path)
    extractor = NewResourceTextExtractor(provider)
    chunker = FixedSizeChunker(config.chunk_size, overlap_size=50)
    embedding_service = HuggingFaceEmbeddingService(
        config.embedding_model_path,
        config.device
    )
    vector_store = FaissVectorStore(embedding_service)

    use_case = IndexResourcesUseCase(
        provider=provider,
        extractor=extractor,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store
    )

    try:
        use_case.execute(str(config.output_index_dir))
        print(f"Success: Index saved to {config.output_index_dir}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()