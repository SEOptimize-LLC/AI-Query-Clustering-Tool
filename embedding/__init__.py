"""
Embedding module for AI Keyword Clustering.
Handles OpenAI embedding generation for keywords.
"""
from embedding.openai_embedder import OpenAIEmbedder, get_openai_embedder
from embedding.batch_processor import BatchEmbeddingProcessor

__all__ = ["OpenAIEmbedder", "get_openai_embedder", "BatchEmbeddingProcessor"]