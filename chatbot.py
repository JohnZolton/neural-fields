"""
Neural Field Chatbot
===================

Interactive chatbot powered by neural field context management.
Allows users to ask questions and receive answers based on the ingested knowledge base.

Usage:
    python chatbot.py --config examples/neural_field_context.yaml --state field_state.json
"""

import os
import argparse
import json
import logging
from typing import List, Tuple, Optional
from datetime import datetime
from neural_field import NeuralField

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NeuralFieldChatbot:
    """Main chatbot class that interfaces with the neural field."""
    
    def __init__(self, config_path: str, state_path: Optional[str] = None):
        """
        Initialize the chatbot.
        
        Args:
            config_path: Path to configuration file
            state_path: Path to saved field state (optional)
        """
        self.config_path = config_path
        self.state_path = state_path
        self.field = NeuralField(config_path)
        
        # Load state if provided
        if state_path and os.path.exists(state_path):
            self._load_state(state_path)
            logger.info(f"Loaded field state from {state_path}")
        
        # Conversation history
        self.conversation_history = []
        self.turn_count = 0
    
    def _load_state(self, state_path: str) -> None:
        """Load field state from JSON file."""
        try:
            with open(state_path, 'r') as f:
                state_data = json.load(f)
            self.field.load_from_dict(state_data)
        except Exception as e:
            logger.error(f"Failed to load state from {state_path}: {e}")
    
    def _save_state(self, state_path: str) -> None:
        """Save field state to JSON file."""
        try:
            with open(state_path, 'w') as f:
                json.dump(self.field.to_dict(), f, indent=2)
            logger.info(f"Saved field state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state to {state_path}: {e}")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and generate a response.
        
        Args:
            query: The user's question or statement
            
        Returns:
            Generated response based on resonant patterns
        """
        self.turn_count += 1
        
        # Inject the query into the field
        query_id = self.field.inject(
            content=f"User query: {query}",
            strength=0.7,
            metadata={
                'type': 'query',
                'turn': self.turn_count,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Get resonant patterns
        resonant_patterns = self.field.get_resonant_patterns(query, threshold=0.2)
        
        # Generate response based on resonant patterns
        response = self._generate_response(query, resonant_patterns)
        
        # Inject the response back into the field
        self.field.inject(
            content=f"Assistant response: {response}",
            strength=0.6,
            metadata={
                'type': 'response',
                'turn': self.turn_count,
                'query_id': query_id,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Record in conversation history
        self.conversation_history.append({
            'turn': self.turn_count,
            'query': query,
            'response': response,
            'resonant_patterns': len(resonant_patterns),
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_response(self, query: str, resonant_patterns: List[Tuple[str, float]]) -> str:
        """
        Generate a response based on resonant patterns.
        
        Args:
            query: The original query
            resonant_patterns: List of (pattern, resonance_strength) tuples
            
        Returns:
            Generated response
        """
        if not resonant_patterns:
            return "I don't have enough information to answer that question. Could you rephrase or provide more context?"
        
        # Sort by resonance strength
        resonant_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Take top patterns
        top_patterns = resonant_patterns[:5]
        
        # Extract relevant content
        relevant_content = []
        for pattern, strength in top_patterns:
            # Clean up the pattern content
            content = pattern.strip()
            if len(content) > 500:
                content = content[:500] + "..."
            relevant_content.append(content)
        
        # Generate response based on relevant content
        if len(relevant_content) == 1:
            response = f"Based on the information I have, here's what I can tell you:\n\n{relevant_content[0]}"
        else:
            response = f"Based on the information I have, here are the key points:\n\n"
            for i, content in enumerate(relevant_content, 1):
                response += f"{i}. {content}\n\n"
        
        return response
    
    def get_field_summary(self) -> dict:
        """Get a summary of the current field state."""
        return self.field.get_field_metrics()
    
    def get_conversation_history(self) -> List[dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        self.turn_count = 0
    
    def interactive_chat(self) -> None:
        """Start an interactive chat session."""
        print("\n" + "="*60)
        print("Neural Field Chatbot")
        print("="*60)
        print("Ask me questions about the ingested knowledge base.")
        print("Type 'quit', 'exit', or 'bye' to end the session.")
        print("Type 'help' for available commands.")
        print("-"*60)
        
        # Show field summary
        summary = self.get_field_summary()
        print(f"\nKnowledge base loaded: {summary['attractor_count']} documents, {summary['pattern_count']} patterns")
        print(f"Field coherence: {summary['coherence']:.3f}, stability: {summary['stability']:.3f}")
        print("-"*60)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'help':
                    self._show_help()
                    continue
                
                if query.lower() == 'summary':
                    self._show_summary()
                    continue
                
                if query.lower() == 'reset':
                    self.reset_conversation()
                    print("Conversation reset.")
                    continue
                
                # Process query
                print("\nThinking...")
                response = self.process_query(query)
                
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, I encountered an error processing your question.")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\nAvailable commands:")
        print("  help     - Show this help message")
        print("  summary  - Show field metrics")
        print("  reset    - Reset conversation history")
        print("  quit     - Exit the chat")
        print("  exit     - Exit the chat")
        print("  bye      - Exit the chat")
    
    def _show_summary(self) -> None:
        """Show field summary."""
        summary = self.get_field_summary()
        print("\nField Summary:")
        print(f"  Patterns: {summary['pattern_count']}")
        print(f"  Attractors: {summary['attractor_count']}")
        print(f"  Coherence: {summary['coherence']:.3f}")
        print(f"  Stability: {summary['stability']:.3f}")
        print(f"  Entropy: {summary['entropy']:.3f}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Neural Field Chatbot')
    parser.add_argument('--config', default='examples/neural_field_context.yaml',
                        help='Path to configuration file')
    parser.add_argument('--state', default='field_state.json',
                        help='Path to saved field state')
    parser.add_argument('--save-state', default='field_state.json',
                        help='File to save field state to')
    
    args = parser.parse_args()
    
    # Create chatbot
    logger.info("Creating neural field chatbot...")
    chatbot = NeuralFieldChatbot(args.config, args.state)
    
    # Start interactive session
    try:
        chatbot.interactive_chat()
    finally:
        # Save state on exit
        chatbot._save_state(args.save_state)


if __name__ == "__main__":
    main()
