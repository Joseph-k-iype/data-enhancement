"""
Business Terms Manager - Core component for managing and matching business terms.

This module provides functionality for storing, retrieving, and matching business terms
using vector similarity search with ChromaDB's HNSW indexing, enhanced with AI evaluation
of term matches.
"""

import csv
import logging
import os
import time
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.db_manager import DBManager
from app.core.embedding import EmbeddingClient, MyDocument
from app.core.models import TaggingResult, TaggingValidationResult # PBT models are in here too
from app.config.environment import get_os_env
from app.config.settings import get_vector_store

# Import agents with error handling
try:
    from app.agents.tagging_evaluation_agent import AITaggingEvaluationAgent
except ImportError:
    AITaggingEvaluationAgent = None # type: ignore
    logging.getLogger(__name__).warning(
        "AITaggingEvaluationAgent not found or failed to import. "
        "AI-based tagging evaluation will not be available."
    )
try:
    from app.agents.term_matching_agent import TermMatchingAgent
except ImportError:
    TermMatchingAgent = None # type: ignore
    logging.getLogger(__name__).warning(
        "TermMatchingAgent not found or failed to import. "
        "Advanced term matching will be limited to basic vector search."
    )


logger = logging.getLogger(__name__)

class BusinessTerm(BaseModel):
    """Model representing a business term in the repository."""
    id: str = Field(..., description="Unique identifier for the term")
    name: str = Field(..., description="Name of the business term")
    description: str = Field(..., description="Description of the business term")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the term")
    
    def dict(self) -> Dict[str, Any]:
        """Convert the business term to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }

class BusinessTermManager:
    """
    Manager for business terms, handling storage, retrieval, and similarity matching.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BusinessTermManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.env = get_os_env()
        self.embedding_client = EmbeddingClient()
        self.db_manager = DBManager()
        self.similarity_threshold = float(self.env.get("SIMILARITY_THRESHOLD", "0.2"))  # Lowered from 0.5 to 0.2
        self.vector_store = get_vector_store()
        self.vector_db_type = self.env.get("VECTOR_DB_TYPE", "chroma").lower()
        
        # Initialize AI evaluation agent
        try:
            from app.agents.tagging_evaluation_agent import AITaggingEvaluationAgent
            self.ai_evaluation_agent = AITaggingEvaluationAgent()
        except ImportError:
            logger.warning("AITaggingEvaluationAgent not available. Using basic evaluation.")
            self.ai_evaluation_agent = None
        
        # Initialize term matching agent
        try:
            from app.agents.term_matching_agent import TermMatchingAgent
            self._term_matching_agent = TermMatchingAgent(self)
        except ImportError:
            logger.warning("TermMatchingAgent not available. Using basic vector search.")
            self._term_matching_agent = None

        logger.info(f"Business term manager initialized with {self.vector_db_type} backend for vectors.")
        logger.info(f"Embedding model in use by EmbeddingClient: {self.embedding_client.embeddings_model}")
        if not self._term_matching_agent:
            logger.warning("TermMatchingAgent is not available; PBT tagging will use basic vector search.")
        if not self.ai_evaluation_agent:
            logger.warning("AITaggingEvaluationAgent is not available; AI evaluation of tagging will be basic.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def import_terms_from_csv(self, csv_path: str, encoding: str = 'utf-8', batch_size: int = 100) -> int:
        try:
            logger.info(f"Starting import from CSV: {csv_path} with encoding: {encoding}, batch size: {batch_size}")
            
            if encoding.lower() in ['auto', 'detect']:
                try:
                    import chardet
                    with open(csv_path, 'rb') as rawfile:
                        file_size = os.path.getsize(csv_path)
                        sample_size = min(1024 * 1024, file_size)
                        sample = rawfile.read(sample_size)
                        detected = chardet.detect(sample)
                        original_encoding_detected = encoding
                        encoding = detected['encoding'] or 'utf-8'
                        logger.info(f"Auto-detected encoding: {encoding} with confidence {detected['confidence']} (user requested: {original_encoding_detected})")
                except Exception as e:
                    logger.warning(f"Encoding auto-detection failed: {e}. Falling back to specified or default UTF-8.")
                    encoding = 'utf-8' if encoding.lower() in ['auto', 'detect'] else encoding

            total_added_count = 0
            current_batch_to_process = []
            processed_row_count = 0

            with open(csv_path, 'r', encoding=encoding, errors='replace') as csvfile:
                reader = csv.DictReader(csvfile)
                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or has no headers.")

                normalized_fieldnames = {fn.lower().replace("_", "").replace(" ", ""): fn for fn in reader.fieldnames}
                
                col_id = normalized_fieldnames.get('id')
                col_pbt_name = normalized_fieldnames.get('pbtname', normalized_fieldnames.get('name'))
                col_pbt_description = normalized_fieldnames.get('pbtdescription', normalized_fieldnames.get('description'))
                col_cdm = normalized_fieldnames.get('cdm')

                if not col_pbt_name or not col_pbt_description:
                    missing_cols = []
                    if not col_pbt_name: missing_cols.append("PBT_NAME or NAME")
                    if not col_pbt_description: missing_cols.append("PBT_DESCRIPTION or DESCRIPTION")
                    raise ValueError(f"CSV file is missing required columns: {', '.join(missing_cols)}. Headers found: {', '.join(reader.fieldnames)}")

                for row_num, row_data in enumerate(reader, 1):
                    processed_row_count +=1
                    pbt_name = row_data.get(col_pbt_name, "").strip()
                    pbt_description = row_data.get(col_pbt_description, "").strip()

                    if not pbt_name or not pbt_description:
                        logger.warning(f"Skipping row {row_num}: PBT_NAME or PBT_DESCRIPTION is empty.")
                        continue

                    term_id = row_data.get(col_id, "").strip() if col_id else ""
                    if not term_id:
                        term_id = f"pbt_{uuid.uuid4()}"
                    
                    metadata = {}
                    if col_cdm and row_data.get(col_cdm):
                        metadata['cdm'] = row_data.get(col_cdm, "").strip()
                    
                    if col_id and row_data.get(col_id, "").strip() and row_data.get(col_id,"").strip() != term_id:
                         metadata['original_csv_id'] = row_data.get(col_id,"").strip()

                    for key, value in row_data.items():
                        original_key_is_main_col = False
                        if col_id and key == col_id: original_key_is_main_col = True
                        if col_pbt_name and key == col_pbt_name: original_key_is_main_col = True
                        if col_pbt_description and key == col_pbt_description: original_key_is_main_col = True
                        if col_cdm and key == col_cdm: original_key_is_main_col = True
                        
                        if not original_key_is_main_col and value and value.strip():
                            meta_key = re.sub(r'[^a-zA-Z0-9_.-]', '_', key.lower()) # ChromaDB metadata key restrictions
                            if len(meta_key) > 63: meta_key = meta_key[:63] # Max key length
                            metadata[meta_key] = value.strip()
                    
                    current_batch_to_process.append({
                        "id": term_id,
                        "name": pbt_name,
                        "description": pbt_description,
                        "metadata": metadata
                    })

                    if len(current_batch_to_process) >= batch_size:
                        logger.info(f"Processing batch of {len(current_batch_to_process)} terms (up to row {row_num})...")
                        batch_added_count = self._process_terms_batch(current_batch_to_process)
                        total_added_count += batch_added_count
                        current_batch_to_process = []
                
                if current_batch_to_process:
                    logger.info(f"Processing final batch of {len(current_batch_to_process)} terms (total rows read: {processed_row_count})...")
                    batch_added_count = self._process_terms_batch(current_batch_to_process)
                    total_added_count += batch_added_count
            
            logger.info(f"CSV import completed. Total rows processed from CSV: {processed_row_count}. Total terms effectively added/updated in vector store: {total_added_count}")
            return total_added_count
        
        except FileNotFoundError:
            logger.error(f"CSV file not found at path: {csv_path}")
            raise IOError(f"CSV file not found: {csv_path}")
        except ValueError as ve:
            logger.error(f"ValueError during CSV import: {ve}")
            raise
        except Exception as e:
            logger.error(f"General error importing terms from CSV: {e}", exc_info=True)
            raise

    def _process_terms_batch(self, terms_batch: List[Dict[str, Any]]) -> int:
        if not terms_batch:
            return 0

        batch_start_time = time.time()
        docs_to_embed = []
        for term_data in terms_batch:
            embedding_text = f"PBT Name: {term_data['name']}. Description: {term_data['description']}"
            if term_data['metadata'].get('cdm'):
                embedding_text += f" CDM: {term_data['metadata']['cdm']}"
            
            docs_to_embed.append(MyDocument(
                id=term_data["id"],
                text=embedding_text,
                metadata=term_data["metadata"]
            ))
        
        logger.debug(f"Generating embeddings for {len(docs_to_embed)} terms in current batch...")
        try:
            docs_with_embeddings = self.embedding_client.batch_generate_embeddings(docs_to_embed)
        except Exception as e:
            logger.error(f"Failed to generate embeddings for a batch: {e}", exc_info=True)
            return 0

        vectors_batch_for_store = []
        for doc_with_embedding in docs_with_embeddings:
            original_term_data = next((t for t in terms_batch if t["id"] == doc_with_embedding.id), None)
            if not original_term_data:
                logger.warning(f"Internal consistency error: Could not find original term data for document ID {doc_with_embedding.id}, skipping.")
                continue

            if not doc_with_embedding.embedding:
                logger.warning(f"Skipping term '{original_term_data['name']}' (ID: {original_term_data['id']}) due to missing embedding.")
                continue
            
            vectors_batch_for_store.append({
                "id": original_term_data["id"],
                "name": original_term_data["name"],
                "description": original_term_data["description"],
                "embedding": doc_with_embedding.embedding,
                "metadata": original_term_data["metadata"] 
            })
        
        inserted_in_this_batch = 0
        if vectors_batch_for_store:
            logger.debug(f"Storing {len(vectors_batch_for_store)} vectors in current batch...")
            try:
                inserted_in_this_batch = self.vector_store.batch_store_vectors(vectors_batch_for_store)
                batch_duration = time.time() - batch_start_time
                logger.info(f"Stored {inserted_in_this_batch} terms from batch in {batch_duration:.2f}s.")
            except Exception as e:
                logger.error(f"Failed to store a batch of vectors: {e}", exc_info=True)
                return 0
        
        return inserted_in_this_batch

    async def tag_element(self, element_id: str, name: str, description: str, 
                    top_k: int = 3, threshold: float = 0.2, # Lowered from 0.3 
                    cdm: Optional[str] = None, 
                    example: Optional[str] = None,
                    process_name: Optional[str] = None,
                    process_description: Optional[str] = None,
                    include_broader_terms: bool = True) -> TaggingResult:
        """
        Find and tag a data element with matching business terms using agentic RAG.
        
        Args:
            element_id: Unique identifier for the element
            name: Element name
            description: Element description
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            cdm: Optional CDM context
            example: Optional example context
            process_name: Optional process name context
            process_description: Optional process description context
            include_broader_terms: Whether to include broader category terms
            
        Returns:
            TaggingResult with matching terms, confidence scores, and metadata
        """
        try:
            if not name or not description:
                logger.warning(f"Empty name or description for element ID: {element_id}. Cannot perform tagging.")
                return TaggingResult(
                    element_id=element_id, element_name=name or "", element_description=description or "",
                    matching_terms=[], confidence_scores=[], modeling_required=True,
                    message="Name or description is empty. Modeling should be performed."
                )
            
            # Use the enhanced TermMatchingAgent with agentic RAG
            if self._term_matching_agent:
                logger.debug(f"Using TermMatchingAgent for element: {element_id}")
                matching_terms_dicts, confidence_scores = await self._term_matching_agent.find_matching_terms(
                    element_id=element_id,
                    element_name=name,
                    element_description=description,
                    top_k=top_k,
                    cdm_context=cdm, 
                    example_context=example,
                    process_name_context=process_name,
                    process_description_context=process_description,
                    initial_threshold=threshold 
                )
                message = "Tagging performed using Agentic TermMatchingAgent."
            else:
                # Fallback to basic vector search if agent is not available
                logger.warning(f"TermMatchingAgent not initialized for element: {element_id}. Using basic search.")
                query_text = f"Item Name: {name}. Description: {description}."
                if example: query_text += f" Example: {example}."
                if process_name: query_text += f" Related Process: {process_name}."
                if process_description: query_text += f" Process Description: {process_description}."
                if cdm: query_text += f" Associated CDM: {cdm}."

                doc = MyDocument(id=element_id, text=query_text)
                doc_with_embedding = self.embedding_client.generate_embeddings(doc)

                if not doc_with_embedding.embedding:
                    logger.error(f"Could not generate embedding for tagging query: {element_id}")
                    return TaggingResult(
                        element_id=element_id, element_name=name, element_description=description,
                        matching_terms=[], confidence_scores=[], modeling_required=True,
                        message="Embedding generation failed. Modeling should be performed."
                    )

                # Use lower threshold for basic vector search
                matching_terms_dicts = self.vector_store.find_similar_vectors(
                    query_vector=doc_with_embedding.embedding,
                    top_k=top_k,
                    threshold=max(0.05, threshold - 0.2) # Use lower threshold
                )
                confidence_scores = [term.get("similarity", 0.0) for term in matching_terms_dicts]
                message = "Tagging performed using basic vector search with reduced threshold."

            # Add match_type to terms if include_broader_terms is True
            if include_broader_terms and matching_terms_dicts:
                # Mark top half as specific and bottom half as broader
                mid_point = max(1, len(matching_terms_dicts) // 2)
                
                for i, term in enumerate(matching_terms_dicts):
                    if i < mid_point:
                        term["match_type"] = "specific"
                    else:
                        term["match_type"] = "broader"
            else:
                # Mark all as specific
                for term in matching_terms_dicts:
                    term["match_type"] = "specific"

            # Determine if modeling is required
            modeling_required = False
            if not matching_terms_dicts:
                modeling_required = True
                message += " No matching terms found."
            elif not confidence_scores or max(confidence_scores, default=0.0) < 0.3:
                modeling_required = True
                message += f" Best match confidence ({max(confidence_scores, default=0.0):.2f}) is below threshold."
            
            if modeling_required:
                message += " Consider modeling a new term."
            else:
                message += f" Found {len(matching_terms_dicts)} relevant PBT(s)."
            
            return TaggingResult(
                element_id=element_id,
                element_name=name,
                element_description=description,
                matching_terms=matching_terms_dicts, 
                confidence_scores=confidence_scores, 
                modeling_required=modeling_required,
                message=message
            )
                
        except Exception as e:
            logger.error(f"Error tagging element '{name}' (ID: {element_id}): {e}", exc_info=True)
            return TaggingResult(
                element_id=element_id, element_name=name, element_description=description,
                matching_terms=[], confidence_scores=[], modeling_required=True,
                message=f"Error during tagging: {str(e)}. Modeling should be performed."
            )

    async def evaluate_tagging_with_reasoning(self, tagging_result: TaggingResult) -> Tuple[float, str]:
        """
        Evaluate the confidence in the tagging with detailed reasoning.
        
        Args:
            tagging_result: Tagging result to evaluate
            
        Returns:
            Tuple containing (confidence_score, reasoning)
        """
        if not self.ai_evaluation_agent:
            logger.warning("AITaggingEvaluationAgent not available for evaluate_tagging_with_reasoning.")
            if not tagging_result.matching_terms: 
                return 0.0, "No terms to evaluate."
            avg_conf = sum(tagging_result.confidence_scores) / len(tagging_result.confidence_scores) if tagging_result.confidence_scores else 0.0
            return avg_conf, "AI evaluation agent not available. Confidence is based on similarity scores."
        
        try:
            if tagging_result.modeling_required and not tagging_result.matching_terms:
                return 0.0, "Modeling is required as no suitable matches were found."
            if not tagging_result.matching_terms:
                return 0.0, "No matching terms were found to evaluate."
            
            is_valid, overall_confidence, reasoning, _ = await self.ai_evaluation_agent.evaluate_tagging_result(tagging_result)
            return overall_confidence, reasoning
        except Exception as e:
            logger.error(f"Error in evaluate_tagging_with_reasoning: {e}")
            return 0.5, f"Error during AI evaluation of tagging: {str(e)}"

    async def validate_tagging(self, tagging_result: TaggingResult) -> TaggingValidationResult:
        """
        Validate a tagging result with AI evaluation.
        
        Args:
            tagging_result: The tagging result to validate
            
        Returns:
            TaggingValidationResult with validation feedback
        """
        if not self.ai_evaluation_agent:
            logger.warning("AITaggingEvaluationAgent not available for validate_tagging.")
            is_valid = bool(tagging_result.matching_terms) and not tagging_result.modeling_required
            feedback = "Tagging seems plausible." if is_valid else "Tagging may require review."
            return TaggingValidationResult(is_valid=is_valid, feedback=feedback, suggested_alternatives=[])
        
        try:
            is_valid, _, reasoning, _ = await self.ai_evaluation_agent.evaluate_tagging_result(tagging_result)
            return TaggingValidationResult(
                is_valid=is_valid,
                feedback=reasoning,
                suggested_alternatives=[]
            )
        except Exception as e:
            logger.error(f"Error in validate_tagging: {e}")
            return TaggingValidationResult(
                is_valid=False, 
                feedback=f"Error during AI validation: {str(e)}", 
                suggested_alternatives=[]
            )
            
    def get_all_terms(self) -> List[BusinessTerm]:
        try:
            term_dicts = self.vector_store.get_all_terms() 
            return [BusinessTerm(**term_dict) for term_dict in term_dicts]
        except Exception as e:
            logger.error(f"Error retrieving all terms: {e}", exc_info=True)
            return []

    def get_term_by_id(self, term_id: str) -> Optional[BusinessTerm]:
        try:
            term_dict = self.vector_store.get_term_by_id(term_id)
            return BusinessTerm(**term_dict) if term_dict else None
        except Exception as e:
            logger.error(f"Error retrieving term by ID '{term_id}': {e}", exc_info=True)
            return None

    def get_term_count(self) -> int:
        try:
            if hasattr(self.vector_store, 'collection') and self.vector_store.collection: # type: ignore
                return self.vector_store.collection.count() # type: ignore
            return len(self.vector_store.get_all_terms())
        except Exception as e:
            logger.error(f"Error getting term count: {e}", exc_info=True)
            return 0

    def delete_term(self, term_id: str) -> bool:
        try:
            return self.vector_store.delete_term(term_id)
        except Exception as e:
            logger.error(f"Error deleting term ID '{term_id}': {e}", exc_info=True)
            return False

    def delete_all_terms(self) -> int:
        try:
            return self.vector_store.delete_all_terms()
        except Exception as e:
            logger.error(f"Error deleting all terms: {e}", exc_info=True)
            return 0

    def search_terms(self, query: str, limit: int = 20) -> List[BusinessTerm]:
        try:
            term_dicts = self.vector_store.search_terms(query, limit)
            return [BusinessTerm(**term_dict) for term_dict in term_dicts]
        except Exception as e:
            logger.error(f"Error searching terms with query '{query}': {e}", exc_info=True)
            return []

    def compute_similarity(self, text1: str, text2: str) -> float:
        try:
            doc1 = MyDocument(id="temp_sim_1", text=text1)
            doc2 = MyDocument(id="temp_sim_2", text=text2)
            
            doc1_embedded = self.embedding_client.generate_embeddings(doc1)
            doc2_embedded = self.embedding_client.generate_embeddings(doc2)

            if not doc1_embedded.embedding or not doc2_embedded.embedding:
                logger.warning("Could not generate embeddings for similarity computation.")
                return 0.0
            
            return self.vector_store.compute_cosine_similarity(
                doc1_embedded.embedding,
                doc2_embedded.embedding
            )
        except Exception as e:
            logger.error(f"Error computing similarity between texts: {e}", exc_info=True)
            return 0.0

    def get_vector_store_info(self) -> Dict[str, Any]:
        try:
            health = self.vector_store.health_check()
            info = {
                "type": self.vector_db_type, 
                "status": health.get("status", "unknown"),
                "term_count": health.get("term_count", self.get_term_count()), 
                "details": health.get("details", {})
            }
            return info
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}", exc_info=True)
            return {"type": self.vector_db_type, "status": "error", "error": str(e)}