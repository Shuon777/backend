"""Logging configuration."""

import sys
import logging


def setup_logging(verbose: bool = False, error_log: str = 'import_errors.log') -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('db_importer.log', encoding='utf-8')
        ]
    )
    
    # Setup error logger
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(error_log, encoding='utf-8')
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)