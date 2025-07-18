"""
Data Ingestion Script
=====================

This script ingests markdown documents from the data/ directory into the neural field,
creating persistent attractors that represent the knowledge base for the chatbot.

Usage:
    python ingest.py --config examples/neural_field_context.yaml
"""

import os
import argparse
import logging
from pathlib import Path
from neural_field import NeuralField

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestor:
    """Handles ingestion of markdown documents into the neural field."""
    
    def __init__(self, neural_field: NeuralField):
        """
        Initialize the data ingestor.
        
        Args:
            neural_field: The neural field to ingest data into
        """
        self.field = neural_field
    
    def ingest_directory(self, directory_path: str, file_pattern: str = "*.md") -> int:
        """
        Ingest all matching files from a directory into the neural field.
        
        Args:
            directory_path: Path to the directory containing files
            file_pattern: File pattern to match (default: "*.md")
            
        Returns:
            Number of files successfully ingested
        """
        directory = Path(directory_path)
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return 0
        
        files = list(directory.glob(file_pattern))
        logger.info(f"Found {len(files)} files to ingest from {directory_path}")
        
        ingested_count = 0
        for file_path in files:
            if self.ingest_file(file_path):
                ingested_count += 1
        
        logger.info(f"Successfully ingested {ingested_count} files")
        return ingested_count
    
    def ingest_file(self, file_path: Path) -> bool:
        """
        Ingest a single markdown file into the neural field.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            True if ingestion was successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from filename
            title = file_path.stem
            
            # Create metadata
            metadata = {
                'source': str(file_path),
                'title': title,
                'type': 'document',
                'file_size': len(content)
            }
            
            # Inject as a strong attractor
            pattern_id = self.field.inject(
                content=content,
                strength=0.9,  # High strength for knowledge base
                metadata=metadata
            )
            
            # Also create a title attractor for easier retrieval
            title_pattern = f"Document: {title}\n{content[:200]}..."  # First 200 chars as summary
            self.field.inject(
                content=title_pattern,
                strength=0.8,
                metadata={'type': 'title', 'source': str(file_path)}
            )
            
            logger.info(f"Ingested {file_path.name} as pattern {pattern_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return False
    
    def ingest_text_content(self, content: str, title: str, source: str = "manual") -> str:
        """
        Ingest arbitrary text content into the neural field.
        
        Args:
            content: The text content to ingest
            title: Title for the content
            source: Source identifier
            
        Returns:
            Pattern ID of the injected content
        """
        metadata = {
            'source': source,
            'title': title,
            'type': 'content',
            'content_length': len(content)
        }
        
        pattern_id = self.field.inject(
            content=content,
            strength=0.85,
            metadata=metadata
        )
        
        logger.info(f"Ingested content '{title}' as pattern {pattern_id}")
        return pattern_id
    
    def get_ingestion_summary(self) -> dict[str, any]:
        """Get a summary of the current field state."""
        metrics = self.field.get_field_metrics()
        return {
            'total_patterns': metrics['pattern_count'] + metrics['attractor_count'],
            'attractors': metrics['attractor_count'],
            'patterns': metrics['pattern_count'],
            'coherence': metrics['coherence'],
            'stability': metrics['stability'],
            'entropy': metrics['entropy']
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Ingest data into neural field')
    parser.add_argument('--config', default='examples/neural_field_context.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing data files')
    parser.add_argument('--pattern', default='*.md',
                        help='File pattern to match')
    parser.add_argument('--save-state', default='field_state.json',
                        help='File to save field state to')
    
    args = parser.parse_args()
    
    # Create neural field
    logger.info("Creating neural field...")
    field = NeuralField(args.config)
    
    # Create ingestor
    ingestor = DataIngestor(field)
    
    # Ingest data directory
    logger.info("Starting data ingestion...")
    ingested_count = ingestor.ingest_directory(args.data_dir, args.pattern)
    
    # Save field state
    logger.info("Saving field state...")
    with open(args.save_state, 'w') as f:
        import json
        json.dump(field.to_dict(), f, indent=2)
    
    # Print summary
    summary = ingestor.get_ingestion_summary()
    logger.info("Ingestion complete!")
    logger.info(f"Files ingested: {ingested_count}")
    logger.info(f"Total patterns: {summary['total_patterns']}")
    logger.info(f"Attractors: {summary['attractors']}")
    logger.info(f"Coherence: {summary['coherence']:.3f}")
    logger.info(f"Stability: {summary['stability']:.3f}")
    
    return ingested_count


if __name__ == "__main__":
    main()
