"""
Custom exceptions for AI Keyword Clustering.
"""


class ClusteringError(Exception):
    """Base exception for clustering errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class APIError(ClusteringError):
    """Exception for API-related errors."""
    
    def __init__(
        self,
        message: str,
        service: str,
        status_code: int = None,
        response: str = None
    ):
        self.service = service
        self.status_code = status_code
        self.response = response
        super().__init__(
            message,
            {
                "service": service,
                "status_code": status_code,
                "response": response
            }
        )


class RateLimitError(APIError):
    """Exception for rate limit errors."""
    
    def __init__(
        self,
        service: str,
        retry_after: int = None
    ):
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {service}",
            service=service,
            status_code=429
        )


class ValidationError(ClusteringError):
    """Exception for data validation errors."""
    
    def __init__(self, message: str, field: str = None, value=None):
        self.field = field
        self.value = value
        super().__init__(
            message,
            {"field": field, "value": str(value) if value else None}
        )


class ConfigurationError(ClusteringError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, missing_keys: list = None):
        self.missing_keys = missing_keys or []
        super().__init__(
            message,
            {"missing_keys": missing_keys}
        )


class DatabaseError(ClusteringError):
    """Exception for database-related errors."""
    
    def __init__(self, message: str, operation: str = None, table: str = None):
        self.operation = operation
        self.table = table
        super().__init__(
            message,
            {"operation": operation, "table": table}
        )


class ProcessingError(ClusteringError):
    """Exception for processing pipeline errors."""
    
    def __init__(
        self,
        message: str,
        phase: str,
        progress: float = 0.0,
        recoverable: bool = True
    ):
        self.phase = phase
        self.progress = progress
        self.recoverable = recoverable
        super().__init__(
            message,
            {
                "phase": phase,
                "progress": progress,
                "recoverable": recoverable
            }
        )


class EmbeddingError(APIError):
    """Exception for embedding generation errors."""
    
    def __init__(self, message: str, batch_index: int = None):
        self.batch_index = batch_index
        super().__init__(
            message,
            service="OpenAI Embeddings"
        )


class LabelingError(APIError):
    """Exception for LLM labeling errors."""
    
    def __init__(self, message: str, model: str = None):
        self.model = model
        super().__init__(
            message,
            service="OpenRouter LLM"
        )