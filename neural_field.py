"""
Neural Field Implementation
==========================

This module implements a neural field for context management in large language models.
It treats context as a continuous medium rather than discrete tokens, allowing for
more fluid and persistent information management through resonance and attractor dynamics.

Based on the concepts from:
- 08_neural_fields_foundations.md
- 09_persistence_and_resonance.md
- 10_field_orchestration.md
- 11_emergence_and_attractor_dynamics.md
"""

import json
import yaml
import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResonanceMeasurer:
    """Measures semantic resonance between text patterns."""
    
    def __init__(self, method: str = 'cosine'):
        """
        Initialize the resonance measurer.
        
        Args:
            method: The method to use for measuring resonance ('cosine', 'jaccard', 'overlap')
        """
        self.method = method
    
    def measure(self, text1: str, text2: str) -> float:
        """
        Measure resonance between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Resonance score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        if self.method == 'cosine':
            return self._cosine_similarity(text1, text2)
        elif self.method == 'jaccard':
            return self._jaccard_similarity(text1, text2)
        elif self.method == 'overlap':
            return self._overlap_coefficient(text1, text2)
        else:
            return self._simple_similarity(text1, text2)
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between texts."""
        words1 = self._get_word_freq(text1.lower())
        words2 = self._get_word_freq(text2.lower())
        
        # Get all unique words
        all_words = set(words1.keys()) | set(words2.keys())
        
        # Create vectors
        vec1 = [words1.get(word, 0) for word in all_words]
        vec2 = [words2.get(word, 0) for word in all_words]
        
        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _overlap_coefficient(self, text1: str, text2: str) -> float:
        """Calculate overlap coefficient between texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        min_size = min(len(words1), len(words2))
        
        return intersection / min_size if min_size > 0 else 0.0
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_word_freq(self, text: str) -> Dict[str, int]:
        """Get word frequency dictionary from text."""
        words = text.lower().split()
        freq = defaultdict(int)
        for word in words:
            freq[word] += 1
        return freq


class Pattern:
    """Represents a semantic pattern in the neural field."""
    
    def __init__(self, content: str, strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a pattern.
        
        Args:
            content: The semantic content of the pattern
            strength: Initial strength of the pattern (0.0 to 1.0)
            metadata: Additional metadata about the pattern
        """
        self.content = content
        self.strength = max(0.0, min(1.0, strength))
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for this pattern."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"pattern_{content_hash}_{int(self.created_at.timestamp())}"
    
    def decay(self, decay_rate: float) -> None:
        """Apply decay to this pattern."""
        self.strength *= (1 - decay_rate)
        self.strength = max(0.0, self.strength)
    
    def amplify(self, factor: float) -> None:
        """Amplify this pattern's strength."""
        self.strength = min(1.0, self.strength * (1 + factor))
        self.last_accessed = datetime.now()
    
    def access(self) -> None:
        """Mark this pattern as accessed."""
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'content': self.content,
            'strength': self.strength,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'id': self.id
        }


class Attractor:
    """Represents a stable attractor in the neural field."""
    
    def __init__(self, pattern: Pattern, basin_width: float = 0.5, resonance_measurer: Optional[ResonanceMeasurer] = None):
        """
        Initialize an attractor.
        
        Args:
            pattern: The pattern that forms this attractor
            basin_width: How broadly this attractor influences the field
            resonance_measurer: The resonance measurer to use for similarity calculations
        """
        self.pattern = pattern
        self.basin_width = max(0.0, min(1.0, basin_width))
        self.formation_time = datetime.now()
        self.influence_count = 0
        self.resonance_measurer = resonance_measurer or ResonanceMeasurer()
    
    def influence(self, pattern: Pattern) -> float:
        """Calculate influence on a given pattern."""
        similarity = self._calculate_similarity(self.pattern.content, pattern.content)
        influence = similarity * self.basin_width * self.pattern.strength
        return influence
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return self.resonance_measurer.measure(text1, text2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert attractor to dictionary."""
        return {
            'pattern': self.pattern.to_dict(),
            'basin_width': self.basin_width,
            'formation_time': self.formation_time.isoformat(),
            'influence_count': self.influence_count
        }


class NeuralField:
    """Main neural field implementation for context management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the neural field.
        
        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path)
        
        # Field state
        self.patterns: Dict[str, Pattern] = {}
        self.attractors: Dict[str, Attractor] = {}
        self.symbolic_residue: List[Dict[str, Any]] = []
        
        # Field parameters
        self.decay_rate = self.config['field']['decay_rate']
        self.boundary_permeability = self.config['field']['boundary_permeability']
        self.resonance_bandwidth = self.config['field']['resonance_bandwidth']
        self.attractor_threshold = self.config['field']['attractor_formation_threshold']
        self.max_capacity = self.config['field']['max_capacity']
        
        # Initialize resonance measurer
        self.resonance_measurer = ResonanceMeasurer(
            method=self.config['resonance']['method']
        )
        
        # Initialize attractors from config
        self._initialize_attractors()
        
        # Statistics
        self.iteration_count = 0
        self.last_decay = datetime.now()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Default configuration
        return {
            'field': {
                'decay_rate': 0.05,
                'boundary_permeability': 0.8,
                'resonance_bandwidth': 0.6,
                'attractor_formation_threshold': 0.7,
                'max_capacity': 8000
            },
            'attractors': [],
            'resonance': {
                'method': 'cosine',
                'threshold': 0.2,
                'amplification': 1.2
            },
            'persistence': {
                'attractor_protection': 0.8,
                'overflow_strategy': 'prune_weakest'
            }
        }
    
    def _initialize_attractors(self) -> None:
        """Initialize attractors from configuration."""
        for attractor_config in self.config.get('attractors', []):
            pattern = Pattern(
                content=attractor_config['pattern'],
                strength=attractor_config.get('strength', 0.7)
            )
            attractor = Attractor(
                pattern=pattern,
                basin_width=attractor_config.get('basin_width', 0.5),
                resonance_measurer=self.resonance_measurer
            )
            self.attractors[attractor.pattern.id] = attractor
    
    def inject(self, content: str, strength: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Inject a new pattern into the field.
        
        Args:
            content: The content to inject
            strength: Initial strength of the pattern
            metadata: Additional metadata
            
        Returns:
            ID of the injected pattern
        """
        pattern = Pattern(content, strength, metadata)
        
        # Check for similar patterns to merge
        similar_pattern = self._find_similar_pattern(pattern)
        if similar_pattern:
            # Merge with existing pattern
            similar_pattern.strength = min(1.0, similar_pattern.strength + strength * 0.3)
            similar_pattern.last_accessed = datetime.now()
            return similar_pattern.id
        
        # Add new pattern
        self.patterns[pattern.id] = pattern
        
        # Check if pattern should become an attractor
        if pattern.strength >= self.attractor_threshold:
            self._form_attractor(pattern)
        
        # Apply boundary permeability
        effective_strength = strength * self.boundary_permeability
        pattern.strength = effective_strength
        
        # Handle capacity limits
        self._manage_capacity()
        
        return pattern.id
    
    def _find_similar_pattern(self, pattern: Pattern) -> Optional[Pattern]:
        """Find a similar pattern to merge with."""
        best_match = None
        best_similarity = 0.0
        
        for existing_pattern in self.patterns.values():
            similarity = self._calculate_similarity(pattern.content, existing_pattern.content)
            if similarity > 0.85 and similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_pattern
        
        return best_match
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return self.resonance_measurer.measure(text1, text2)
    
    def _form_attractor(self, pattern: Pattern) -> None:
        """Form a new attractor from a pattern."""
        if pattern.id not in [a.pattern.id for a in self.attractors.values()]:
            attractor = Attractor(pattern, basin_width=0.5, resonance_measurer=self.resonance_measurer)
            self.attractors[pattern.id] = attractor
            logger.debug(f"Formed new attractor: {pattern.id}")
    
    def _manage_capacity(self) -> None:
        """Manage field capacity by removing weak patterns."""
        total_patterns = len(self.patterns) + len(self.attractors)
        if total_patterns <= self.max_capacity:
            return
        
        # Remove weakest patterns first
        strategy = self.config['persistence']['overflow_strategy']
        
        if strategy == 'prune_weakest':
            # Remove weakest non-attractor patterns
            patterns_by_strength = sorted(
                [p for p in self.patterns.values() if p.id not in self.attractors],
                key=lambda p: p.strength
            )
            
            to_remove = min(len(patterns_by_strength), total_patterns - self.max_capacity)
            for pattern in patterns_by_strength[:to_remove]:
                del self.patterns[pattern.id]
    
    def decay(self) -> None:
        """Apply natural decay to all patterns."""
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_decay).total_seconds()
        
        # Apply decay based on time elapsed
        decay_factor = self.decay_rate * min(1.0, time_elapsed / 3600)  # Normalize to hours
        
        # Decay patterns
        patterns_to_remove = []
        for pattern_id, pattern in self.patterns.items():
            pattern.decay(decay_factor)
            if pattern.strength < 0.01:
                patterns_to_remove.append(pattern_id)
        
        # Remove decayed patterns
        for pattern_id in patterns_to_remove:
            del self.patterns[pattern_id]
        
        # Decay attractors (slower decay)
        attractors_to_remove = []
        for attractor_id, attractor in self.attractors.items():
            attractor.pattern.decay(decay_factor * 0.2)  # Slower decay for attractors
            if attractor.pattern.strength < 0.1:
                attractors_to_remove.append(attractor_id)
        
        # Remove decayed attractors
        for attractor_id in attractors_to_remove:
            del self.attractors[attractor_id]
        
        self.last_decay = current_time
    
    def measure_resonance(self, pattern1: str, pattern2: str) -> float:
        """Measure resonance between two patterns."""
        return self._calculate_similarity(pattern1, pattern2)
    
    def get_resonant_patterns(self, query: str, threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Get patterns that resonate with a query."""
        resonant_patterns = []
        
        # Check patterns
        for pattern in self.patterns.values():
            resonance = self.measure_resonance(query, pattern.content)
            if resonance >= threshold:
                resonant_patterns.append((pattern.content, resonance * pattern.strength))
        
        # Check attractors
        for attractor in self.attractors.values():
            resonance = self.measure_resonance(query, attractor.pattern.content)
            if resonance >= threshold:
                influence = resonance * attractor.pattern.strength * attractor.basin_width
                resonant_patterns.append((attractor.pattern.content, influence))
        
        # Sort by resonance strength
        resonant_patterns.sort(key=lambda x: x[1], reverse=True)
        return resonant_patterns
    
    def get_field_metrics(self) -> Dict[str, float]:
        """Get comprehensive metrics about the field."""
        self.decay()  # Apply decay before measuring
        
        # Calculate basic metrics
        pattern_count = len(self.patterns)
        attractor_count = len(self.attractors)
        
        # Calculate average pattern strength
        avg_pattern_strength = 0.0
        if self.patterns:
            avg_pattern_strength = sum(p.strength for p in self.patterns.values()) / len(self.patterns)
        
        # Calculate average attractor strength
        avg_attractor_strength = 0.0
        if self.attractors:
            avg_attractor_strength = sum(a.pattern.strength for a in self.attractors.values()) / len(self.attractors)
        
        # Calculate coherence (simplified)
        coherence = 0.0
        if self.attractors and self.patterns:
            total_resonance = 0.0
            for pattern in self.patterns.values():
                max_resonance = 0.0
                for attractor in self.attractors.values():
                    resonance = self.measure_resonance(pattern.content, attractor.pattern.content)
                    max_resonance = max(max_resonance, resonance)
                total_resonance += max_resonance * pattern.strength
            
            total_strength = sum(p.strength for p in self.patterns.values())
            if total_strength > 0:
                coherence = total_resonance / total_strength
        
        # Calculate stability
        stability = 0.0
        if self.attractors:
            stability = avg_attractor_strength
        
        # Calculate entropy
        entropy = 0.0
        if self.patterns:
            total_strength = sum(p.strength for p in self.patterns.values())
            if total_strength > 0:
                probabilities = [p.strength / total_strength for p in self.patterns.values()]
                entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
                entropy = entropy / math.log2(len(self.patterns)) if len(self.patterns) > 1 else 0.0
        
        return {
            'pattern_count': pattern_count,
            'attractor_count': attractor_count,
            'avg_pattern_strength': avg_pattern_strength,
            'avg_attractor_strength': avg_attractor_strength,
            'coherence': coherence,
            'stability': stability,
            'entropy': entropy,
            'total_patterns': pattern_count + attractor_count
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire field to dictionary."""
        return {
            'patterns': {pid: pattern.to_dict() for pid, pattern in self.patterns.items()},
            'attractors': {aid: attractor.to_dict() for aid, attractor in self.attractors.items()},
            'config': self.config,
            'metrics': self.get_field_metrics(),
            'iteration_count': self.iteration_count
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load field state from dictionary."""
        # Load patterns
        self.patterns = {}
        for pid, pattern_data in data.get('patterns', {}).items():
            pattern = Pattern(
                content=pattern_data['content'],
                strength=pattern_data['strength'],
                metadata=pattern_data.get('metadata', {})
            )
            pattern.created_at = datetime.fromisoformat(pattern_data['created_at'])
            pattern.last_accessed = datetime.fromisoformat(pattern_data['last_accessed'])
            self.patterns[pid] = pattern
        
        # Load attractors
        self.attractors = {}
        for aid, attractor_data in data.get('attractors', {}).items():
            pattern_data = attractor_data['pattern']
            pattern = Pattern(
                content=pattern_data['content'],
                strength=pattern_data['strength'],
                metadata=pattern_data.get('metadata', {})
            )
            pattern.created_at = datetime.fromisoformat(pattern_data['created_at'])
            pattern.last_accessed = datetime.fromisoformat(pattern_data['last_accessed'])
            attractor = Attractor(
                pattern=pattern,
                basin_width=attractor_data['basin_width'],
                resonance_measurer=self.resonance_measurer
            )
            self.attractors[aid] = attractor
        
        logger.info(f"Loaded field with {len(self.patterns)} patterns and {len(self.attractors)} attractors")


# Example usage
if __name__ == "__main__":
    # Create a neural field with default configuration
    field = NeuralField()
    
    # Inject some patterns
    field.inject("Neural fields treat context as a continuous medium.")
    field.inject("Information persists through resonance rather than explicit storage.")
    field.inject("Patterns that align with existing field structures decay more slowly.")
    
    # Get metrics
    metrics = field.get_field_metrics()
    print("Field metrics:", metrics)
    
    # Find resonant patterns
    resonant = field.get_resonant_patterns("neural field context")
    print("Resonant patterns:", resonant[:3])
