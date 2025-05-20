"""
Embedding Client for generating vector embeddings using Azure OpenAI.
With fallback to local embedding generation when Azure authentication fails.
"""

import logging
import numpy as np
import hashlib
import json
import os
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from app.config.environment import get_os_env
from utils.auth_helper import get_azure_token_cached
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

"""
Embedding Client for generating vector embeddings using Azure OpenAI.
With fallback to local embedding generation when Azure authentication fails.
"""

import logging
import numpy as np
import hashlib
import json
import os
import time
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
from pydantic import BaseModel
from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class MyDocument(BaseModel):
    """Model representing a document with its embedding."""
    id: str = ""
    text: str = ""
    embedding: List[float] = []
    metadata: Dict[str, Any] = {}


class EmbeddingClient:
    """Client for generating embeddings for documents using Azure OpenAI."""
    
    # Define model dimension mappings for reference only (not used for limiting)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    class EmbeddingClient:
        """Client for generating embeddings for documents using Azure OpenAI."""
        
        # Define model dimension mappings for reference only (not used for limiting)
        MODEL_DIMENSIONS = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
    def __init__(self, env: Optional[Any] = None, azure_api_version: str = "2023-05-15", embeddings_model: str = None):
        """
        Initialize the embedding client.
        
        Args:
            env: OSEnv instance (optional)
            azure_api_version: API version for Azure OpenAI
            embeddings_model: Model to use for embeddings
        """
        if env is None:
            # Import here to avoid circular imports
            from app.config.environment import get_os_env
            self.env = get_os_env()
        else:
            self.env = env
            
        self.azure_api_version = azure_api_version
        
        # Get embedding model from environment or use default
        self.embeddings_model = embeddings_model or self.env.get("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Path for cache directory
        self.cache_dir = os.path.join(os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db"), "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"Initializing embedding client with model: {self.embeddings_model}")
        
        try:
            self.direct_azure_client = self._get_direct_azure_client()
            self.use_azure = True
            logger.info(f"Embedding client initialized with Azure OpenAI model: {self.embeddings_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI client: {e}. Using local embedding generation.")
            self.direct_azure_client = None
            self.use_azure = False
    
    def _get_direct_azure_client(self):
        """Get the Azure OpenAI client for generating embeddings with token caching."""
        try:
            # Get tenant, client and secret info for token acquisition
            tenant_id = self.env.get("AZURE_TENANT_ID", "")
            client_id = self.env.get("AZURE_CLIENT_ID", "")
            client_secret = self.env.get("AZURE_CLIENT_SECRET", "")
            azure_endpoint = self.env.get("AZURE_ENDPOINT", "")
            
            # Get cached token or generate a new one
            try:
                # First try to use the cached token helper if available
                from utils.auth_helper import get_azure_token_cached
                token = get_azure_token_cached(
                    tenant_id=tenant_id,
                    client_id=client_id, 
                    client_secret=client_secret,
                    scope="https://cognitiveservices.azure.com/.default"
                )
                logger.info("Successfully obtained Azure token using cached token helper")
            except ImportError:
                # Fall back to direct token acquisition
                logger.warning("Token caching helper not available. Using direct token acquisition.")
                from azure.identity import ClientSecretCredential
                credential = ClientSecretCredential(
                    tenant_id=tenant_id,
                    client_id=client_id,
                    client_secret=client_secret
                )
                token = credential.get_token("https://cognitiveservices.azure.com/.default").token
                logger.info("Successfully obtained Azure token using direct acquisition")
            
            if not token:
                logger.error("Failed to obtain Azure token")
                raise ValueError("Failed to obtain Azure token")
                
            # Create client with token as API key
            return AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=self.azure_api_version,
                api_key=token  # Use token as API key
            )
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise
    
    def _generate_local_embedding(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding locally when Azure OpenAI is unavailable.
        This is a fallback method that creates consistent but less semantically meaningful embeddings.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding
        """
        # Check if we have a cached embedding for this text
        cache_key = self._get_cache_key(text)
        cached_embedding = self._get_cached_embedding(cache_key)
        
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return cached_embedding
        
        logger.info(f"Generating local embedding for text: {text[:50]}...")
        
        # Create a hash of the text
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Use the hash to seed a random number generator
        np.random.seed(int.from_bytes(text_hash[:4], byteorder='big'))
        
        # Generate a random embedding with the default dimension of 1536
        # but this is just a default, not enforced as a limit
        embedding = np.random.normal(0, 1, 1536).tolist()
        
        # Normalize the embedding to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        # Save to cache
        self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for a text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{text_hash}.json"
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[List[float]]:
        """Get an embedding from the cache if it exists."""
        cache_path = os.path.join(self.cache_dir, cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return data.get('embedding')
            except Exception as e:
                logger.warning(f"Error reading cached embedding: {e}")
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Save an embedding to the cache."""
        cache_path = os.path.join(self.cache_dir, cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({'embedding': embedding}, f)
        except Exception as e:
            logger.warning(f"Error caching embedding: {e}")
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced retries to fail faster
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((Exception)),
        reraise=True
    )
    def generate_embeddings(self, doc: MyDocument, auto_reduce: bool = False) -> MyDocument:
        """
        Generate embeddings for a document with retry logic.
        
        Args:
            doc: Document to generate embeddings for
            auto_reduce: Whether to automatically reduce dimensions (ignored - kept for API compatibility)
            
        Returns:
            Document with embedding
        """
        try:
            # Check if text is empty
            if not doc.text or len(doc.text.strip()) == 0:
                logger.warning(f"Empty text for document ID: {doc.id}")
                return doc
            
            # Use Azure OpenAI if available
            if self.use_azure and self.direct_azure_client:
                try:
                    # Check local cache for this text first
                    cache_key = self._get_cache_key(doc.text)
                    cached_embedding = self._get_cached_embedding(cache_key)
                    if cached_embedding is not None:
                        logger.debug(f"Using cached embedding for '{doc.id}' from file cache")
                        doc.embedding = cached_embedding
                        return doc
                    
                    # Try to generate embedding using Azure OpenAI
                    start_time = time.time()
                    try:
                        response = self.direct_azure_client.embeddings.create(
                            model=self.embeddings_model,
                            input=doc.text
                        ).data[0].embedding
                    except Exception as token_error:
                        # Check if this is a token expiration error
                        if "401" in str(token_error) or "unauthorized" in str(token_error).lower():
                            logger.warning("Token appears to be expired. Refreshing token and retrying...")
                            # Refresh token and recreate the client
                            self._refresh_token()
                            
                            # Try again with new token
                            response = self.direct_azure_client.embeddings.create(
                                model=self.embeddings_model,
                                input=doc.text
                            ).data[0].embedding
                        else:
                            # Not a token error, re-raise
                            raise
                    
                    # Log the dimension of the embedding for debugging and performance
                    original_dim = len(response)
                    generation_time = time.time() - start_time
                    logger.debug(f"Generated Azure embedding for '{doc.id}' with dimension: {original_dim} in {generation_time:.2f}s")
                    
                    # Use the embedding as-is, no dimension reduction
                    doc.embedding = response
                    
                    # Cache the embedding for future use
                    self._cache_embedding(cache_key, response)
                    
                except Exception as azure_error:
                    logger.warning(f"Azure embedding generation failed: {azure_error}. Using local fallback.")
                    doc.embedding = self._generate_local_embedding(doc.text)
            else:
                # Use local embedding generation
                doc.embedding = self._generate_local_embedding(doc.text)
            
            return doc
        except Exception as e:
            logger.error(f"Error generating embeddings (attempt will be retried): {e}")
            raise

    def _refresh_token(self):
        """Refresh the Azure token and recreate the client."""
        try:
            logger.info("Refreshing Azure token...")
            # Get tenant, client and secret info for token acquisition
            tenant_id = self.env.get("AZURE_TENANT_ID", "")
            client_id = self.env.get("AZURE_CLIENT_ID", "")
            client_secret = self.env.get("AZURE_CLIENT_SECRET", "")
            
            # Create a fresh credential and get a new token
            from azure.identity import ClientSecretCredential
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            
            token = credential.get_token("https://cognitiveservices.azure.com/.default").token
            
            # Update the client with the new token
            self.direct_azure_client = AzureOpenAI(
                azure_endpoint=self.env.get("AZURE_ENDPOINT", ""),
                api_version=self.azure_api_version,
                api_key=token  # Use token as API key
            )
            
            logger.info("Azure token refreshed successfully")
        except Exception as e:
            logger.error(f"Error refreshing Azure token: {e}")
            raise
            
    def generate_synonyms(self, term_name: str, term_definition: str, max_synonyms: int = 10) -> List[str]:
        """
        Generate synonyms for a business term using the LLM.
        
        Args:
            term_name: Name of the business term
            term_definition: Definition of the business term
            max_synonyms: Maximum number of synonyms to generate
            
        Returns:
            List of synonyms
        """
        try:
            # Create the prompt for synonym generation
            prompt = f"""
            Generate {max_synonyms} alternative terms, phrases or synonyms that business users might use when referring to this business term:
            
            Term: {term_name}
            Definition: {term_definition}
            
            Provide ONLY a comma-separated list of alternative terms or phrases that a user might use when referring to this concept.
            These should be different ways to express the same concept, including industry jargon, abbreviations, and common variations.
            DO NOT provide explanations - ONLY the comma-separated list of terms.
            """
            
            # Use Azure OpenAI Completion API
            try:
                # Try to call the API with current token
                try:
                    response = self.direct_azure_client.chat.completions.create(
                        model=self.env.get("MODEL_NAME", "gpt-4o"),
                        messages=[
                            {"role": "system", "content": "You are an expert in business terminology."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                except Exception as token_error:
                    # Check if this is a token expiration error
                    if "401" in str(token_error) or "unauthorized" in str(token_error).lower():
                        logger.warning("Token appears to be expired when generating synonyms. Refreshing token and retrying...")
                        # Refresh token and recreate the client
                        self._refresh_token()
                        
                        # Try again with new token
                        response = self.direct_azure_client.chat.completions.create(
                            model=self.env.get("MODEL_NAME", "gpt-4o"),
                            messages=[
                                {"role": "system", "content": "You are an expert in business terminology."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=300
                        )
                    else:
                        # Not a token error, re-raise
                        raise
                
                # Extract the synonyms from the response
                synonyms_text = response.choices[0].message.content.strip()
                
                # Split by comma and clean up each synonym
                synonyms = [syn.strip() for syn in synonyms_text.split(',')]
                
                # Remove duplicates and empty strings
                synonyms = list(set([syn for syn in synonyms if syn]))
                
                logger.info(f"Generated {len(synonyms)} synonyms for '{term_name}'")
                return synonyms
                
            except Exception as e:
                logger.error(f"Error using Azure OpenAI to generate synonyms: {e}")
                # Fall back to a simple algorithmic approach if API call fails
                return self._generate_simple_synonyms(term_name)
                
        except Exception as e:
            logger.error(f"Error generating synonyms for '{term_name}': {e}")
            return self._generate_simple_synonyms(term_name)
    
    def _generate_simple_synonyms(self, term_name: str) -> List[str]:
        """
        Generate simple synonyms algorithmically when API calls fail.
        
        Args:
            term_name: Name of the business term
            
        Returns:
            List of basic synonyms
        """
        # Generate basic variations
        synonyms = []
        
        # Add the original term
        synonyms.append(term_name)
        
        # Add lowercase and uppercase variations
        synonyms.append(term_name.lower())
        synonyms.append(term_name.upper())
        
        # Add abbreviation if term has multiple words
        words = term_name.split()
        if len(words) > 1:
            abbreviation = ''.join(word[0] for word in words if word).upper()
            synonyms.append(abbreviation)
        
        # Add variations with common prefixes/suffixes
        if not term_name.lower().startswith("the "):
            synonyms.append(f"The {term_name}")
        
        # Add CamelCase and snake_case variations
        if ' ' in term_name:
            camel_case = ''.join(word.capitalize() for word in words)
            snake_case = '_'.join(word.lower() for word in words)
            synonyms.append(camel_case)
            synonyms.append(snake_case)
        
        # Remove duplicates and empty strings
        synonyms = list(set([syn for syn in synonyms if syn]))
        
        logger.info(f"Generated {len(synonyms)} simple synonyms for '{term_name}'")
        return synonyms
    
    def batch_generate_embeddings(self, docs: List[MyDocument], batch_size: int = 20, auto_reduce: bool = False) -> List[MyDocument]:
        """
        Generate embeddings for multiple documents in batches.
        
        Args:
            docs: List of documents to generate embeddings for
            batch_size: Number of documents to process in each batch
            auto_reduce: Whether to automatically reduce dimensions (ignored - kept for API compatibility)
            
        Returns:
            List of documents with embeddings
        """
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            
            for doc in batch:
                try:
                    doc_with_embedding = self.generate_embeddings(doc, auto_reduce=False)
                    results.append(doc_with_embedding)
                except Exception as e:
                    logger.error(f"Error generating embedding for document {doc.id}: {e}")
                    # Add the document without embedding
                    results.append(doc)
        
        return results
    
    # Kept for API compatibility but not used for dimension reduction
    def reduce_dimensions(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        This method is kept for API compatibility but returns the original embedding.
        No dimension reduction is performed.
        
        Args:
            embedding: Original embedding vector
            target_dim: Target dimension for the reduced vector (ignored)
            
        Returns:
            Original embedding vector unchanged
        """
        logger.debug("Dimension reduction requested but disabled - returning original embedding")
        return embedding
    
    # Kept for API compatibility but not used
    def expand_dimensions(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        This method is kept for API compatibility but returns the original embedding.
        
        Args:
            embedding: Original embedding vector
            target_dim: Target dimension for the expanded vector (ignored)
            
        Returns:
            Original embedding vector unchanged
        """
        logger.debug("Dimension expansion requested but disabled - returning original embedding")
        return embedding
    
    # Kept for API compatibility but not used
    def adjust_embedding_dimension(self, embedding: List[float], target_dim: int) -> List[float]:
        """
        This method is kept for API compatibility but returns the original embedding.
        
        Args:
            embedding: Original embedding vector
            target_dim: Target dimension (ignored)
        
        Returns:
            Original embedding vector unchanged
        """
        logger.debug("Embedding dimension adjustment requested but disabled - returning original embedding")
        return embedding
    
    # Alias method with more descriptive name - supports both naming conventions
    def generate_embeddings_for_document(self, doc: MyDocument, auto_reduce: bool = False) -> MyDocument:
        """Alias for generate_embeddings."""
        return self.generate_embeddings(doc, auto_reduce=False)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute the cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        # Make sure embeddings have same dimension
        min_dim = min(len(embedding1), len(embedding2))
        if len(embedding1) != len(embedding2):
            logger.warning(f"Embeddings have different dimensions: {len(embedding1)} vs {len(embedding2)}. Using first {min_dim} dimensions for comparison.")
            embedding1 = embedding1[:min_dim]
            embedding2 = embedding2[:min_dim]
            
        # Convert to numpy arrays
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        similarity = dot_product / (norm_a * norm_b)
        return max(0.0, min(similarity, 1.0))  # Ensure in range [0, 1]