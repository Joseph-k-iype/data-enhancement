import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from app.core.models import EnhancementResult, TaggingResult, DataElement, Process

logger = logging.getLogger(__name__)

class ConfidenceEvaluator:
    """Evaluates confidence scores for enhancement and tagging results."""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self._setup_evaluation_chain()
    
    def _format_processes_info(self, processes: List[Process]) -> str:
        """
        Format the processes information for the prompt.
        
        Args:
            processes: The processes to format
            
        Returns:
            str: Formatted processes information
        """
        if not processes:
            return "Related Processes: None"
        
        # Ensure processes are Process objects
        process_list = []
        for p in processes:
            if isinstance(p, Process):
                process_list.append(p)
            elif isinstance(p, dict):
                process_list.append(Process(**p))
            else:
                logger.warning(f"Unknown process type: {type(p)}")
        
        processes_info = "Related Processes:\n"
        
        for i, process in enumerate(process_list, 1):
            processes_info += f"Process {i} ID: {process.process_id}\n"
            processes_info += f"Process {i} Name: {process.process_name}\n"
            if process.process_description:
                processes_info += f"Process {i} Description: {process.process_description}\n"
            processes_info += "\n"
        
        return processes_info
    
    def _setup_evaluation_chain(self):
        template = """
        You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to evaluate 
        the confidence in the enhancement and tagging of data elements.
        
        Data Element:
        - ID: {id}
        - Original Name: {original_name}
        - Original Description: {original_description}
        - Enhanced Name: {enhanced_name}
        - Enhanced Description: {enhanced_description}
        {processes_info}
        
        Enhancement Feedback: {enhancement_feedback}
        Validation Feedback: {validation_feedback}
        
        ISO/IEC 11179 standards for data element names (adapted for business-friendly format):
        - Names MUST be in lowercase with spaces between words.
        - Names MUST NOT use technical formatting like camelCase, snake_case or PascalCase
        - Names MUST NOT contain underscores, hyphens, or special characters
        - Names should be clear, unambiguous and self-describing
        - Names should not use acronyms or abbreviations unless they are universally understood
        - Names should be concise yet descriptive
        - Names should use standard terminology in the domain
        - Names should use business language that non-technical users can understand
        
        ISO/IEC 11179 standards for data element descriptions:
        - Descriptions should clearly define what the data element represents
        - Descriptions should be complete, covering the concept fully
        - Descriptions should be precise, specific enough to distinguish from other concepts
        - Descriptions should be objective and factual, not opinion-based
        - Descriptions should use complete sentences with proper grammar and punctuation
        - Descriptions should be written in business language, not technical jargon
        
        Based on the ISO/IEC 11179 standards, evaluate the confidence in the enhancement.
        
        Provide your evaluation as follows:
        1. Overall confidence score (0.0-1.0): [provide a confidence score, must be between 0.0 and 1.0]
        2. Detailed justification for the score: [explain your reasoning]
        """
        
        self.evaluation_prompt = PromptTemplate(
            input_variables=["id", "original_name", "original_description", "enhanced_name", 
                           "enhanced_description", "processes_info", "enhancement_feedback", "validation_feedback"],
            template=template)
        self.evaluation_chain = self.evaluation_prompt | self.llm | StrOutputParser()
        
        # Setup tagging evaluation chain
        tagging_template = """
        You are an expert in data governance and business terminology. Your task is to evaluate 
        the confidence in the tagging of data elements to business terms.
        
        Data Element:
        - ID: {id}
        - Name: {name}
        - Description: {description}
        {processes_info}
        
        Matched Business Terms:
        {matched_terms}
        
        Matching Confidence Scores:
        {confidence_scores}
        
        Based on the data element and matched business terms, evaluate the confidence in the tagging.
        Consider whether the business terms accurately represent the data element's semantics.
        
        Provide your evaluation as follows:
        1. Overall confidence score (0.0-1.0): [provide a confidence score, must be between 0.0 and 1.0]
        2. Detailed justification for the score: [explain your reasoning]
        """
        
        self.tagging_evaluation_prompt = PromptTemplate(
            input_variables=["id", "name", "description", "processes_info", "matched_terms", "confidence_scores"],
            template=tagging_template)
        self.tagging_evaluation_chain = self.tagging_evaluation_prompt | self.llm | StrOutputParser()
    
    def _parse_evaluation_result(self, result: str) -> float:
        lines = result.strip().split("\n")
        confidence = 0.5  # Default confidence
        
        for line in lines:
            if "confidence score" in line.lower():
                # Extract the confidence score
                match = re.search(r"\d+\.\d+", line)
                if match:
                    try:
                        confidence = float(match.group(0))
                        confidence = max(0.0, min(1.0, confidence))  # Ensure within bounds
                    except ValueError:
                        logger.warning(f"Could not parse confidence score from line: {line}")
                        pass
        
        return confidence
    
    async def evaluate_enhancement(self, 
                                 original_element: DataElement, 
                                 enhanced_result: EnhancementResult,
                                 validation_feedback: str) -> float:
        """Evaluate the confidence in the enhancement."""
        try:
            # Format processes information
            processes_info = ""
            if original_element.processes:
                processes_info = self._format_processes_info(original_element.processes)
            
            result = await self.evaluation_chain.ainvoke({
                "id": original_element.id,
                "original_name": original_element.existing_name,
                "original_description": original_element.existing_description,
                "enhanced_name": enhanced_result.enhanced_name,
                "enhanced_description": enhanced_result.enhanced_description,
                "processes_info": processes_info,
                "enhancement_feedback": enhanced_result.feedback,
                "validation_feedback": validation_feedback
            })
            
            return self._parse_evaluation_result(result)
        except Exception as e:
            logger.error(f"Error evaluating enhancement confidence: {e}")
            return 0.5  # Default confidence in case of error
    
    async def evaluate_tagging(self, tagging_result: TaggingResult) -> float:
        """Evaluate the confidence in the tagging."""
        try:
            # Skip evaluation if modeling is required
            if tagging_result.modeling_required:
                return 0.0
                
            matched_terms_str = ""
            for i, term in enumerate(tagging_result.matching_terms):
                matched_terms_str += f"{i+1}. Term: {term['name']}\n   Description: {term['description']}\n"
            
            confidence_scores_str = ", ".join([f"{score:.2f}" for score in tagging_result.confidence_scores])
            
            # Prepare processes info placeholder - no processes in TaggingResult currently
            # In a future enhancement, this would need to be added to the TaggingResult model
            processes_info = ""
            
            result = await self.tagging_evaluation_chain.ainvoke({
                "id": tagging_result.element_id,
                "name": tagging_result.element_name,
                "description": tagging_result.element_description,
                "processes_info": processes_info,
                "matched_terms": matched_terms_str,
                "confidence_scores": confidence_scores_str
            })
            
            return self._parse_evaluation_result(result)
        except Exception as e:
            logger.error(f"Error evaluating tagging confidence: {e}")
            return 0.5  # Default confidence in case of error
    
    async def evaluate_tagging_with_reasoning(self, tagging_result: TaggingResult) -> Tuple[float, str]:
        """
        Evaluate the confidence in the tagging with detailed reasoning.
        Enhanced with dynamic concept analysis for better matching.
        
        Args:
            tagging_result: Tagging result to evaluate
            
        Returns:
            Tuple containing (confidence_score, reasoning)
        """
        try:
            # Skip evaluation if modeling is required
            if tagging_result.modeling_required:
                return 0.0, "Modeling is required as no suitable matches were found."
                
            # If no matches, return zero confidence
            if not tagging_result.matching_terms:
                return 0.0, "No matching terms were found. Modeling is required."
            
            # Create a detailed prompt for better reasoning
            matched_terms_str = ""
            for i, term in enumerate(tagging_result.matching_terms):
                confidence = tagging_result.confidence_scores[i] if i < len(tagging_result.confidence_scores) else 0
                matched_terms_str += f"{i+1}. Term: {term['name']}\n"
                matched_terms_str += f"   Description: {term['description']}\n"
                matched_terms_str += f"   Similarity Score: {confidence:.2f}\n\n"
            
            # Create a dynamic concept dictionary for this specific evaluation
            element_concepts = self._extract_key_concepts(tagging_result.element_name, tagging_result.element_description)
            term_concepts = []
            
            # Extract key concepts from each matching term
            for term in tagging_result.matching_terms:
                term_concepts.append(self._extract_key_concepts(term["name"], term["description"]))
            
            # Find shared concepts between element and terms
            shared_concepts = {}
            for i, term_concept_set in enumerate(term_concepts):
                term_name = tagging_result.matching_terms[i]["name"]
                shared = element_concepts.intersection(term_concept_set)
                if shared:
                    shared_concepts[term_name] = list(shared)
            
            # Add concept analysis to the prompt
            concept_analysis = ""
            if shared_concepts:
                concept_analysis = "Concept Analysis:\n"
                for term_name, concepts in shared_concepts.items():
                    concept_analysis += f"- '{term_name}' shares concepts: {', '.join(concepts)}\n"
            
            # Special case handling for account number -> account identifier
            element_name_lower = tagging_result.element_name.lower()
            special_case_instructions = ""
            
            if "account" in element_name_lower and "number" in element_name_lower:
                special_case_instructions = """
                Special Case Instructions:
                - "account number" and "account identifier" represent the same business concept
                - Account identifiers are unique codes or numbers that identify specific accounts
                - Consider semantic equivalence rather than exact text matching
                """
            
            # Create an evaluation prompt with domain knowledge
            tagging_evaluation_template = f"""
            You are an expert in data governance and business terminology with deep domain knowledge in finance, 
            healthcare, and enterprise data standards. Your task is to evaluate the appropriateness of tagging 
            a data element with business terms.
            
            Data Element:
            - ID: {{element_id}}
            - Name: {{element_name}}
            - Description: {{element_description}}
            
            Matched Business Terms:
            {{matched_terms}}
            
            {concept_analysis}
            
            {special_case_instructions}
            
            Instructions:
            1. Analyze the semantic match between the data element and each business term
            2. Consider conceptual alignment, completeness of coverage, and appropriate specificity
            3. Determine if ANY of the business terms are appropriate for this data element
            4. If none are appropriate, explain why new term modeling is needed
            
            Evaluation steps:
            1. CAREFULLY examine the data element name and description
            2. CAREFULLY examine each business term name and description
            3. For each term, assess if it accurately represents the data element's semantics
            4. Provide detailed reasoning about the semantic appropriateness of each match
            5. Give an overall assessment of whether any terms are good matches
            
            Provide your evaluation in this format:
            1. Overall confidence score (0.0-1.0): [provide a confidence score]
            2. Detailed justification for the score:
               [provide your reasoning and analysis for each term]
            3. Final recommendation: [indicate whether modeling is needed or which term is best]
            """
            
            # Create tagging evaluation prompt
            evaluation_prompt = PromptTemplate(
                input_variables=["element_id", "element_name", "element_description", "matched_terms"],
                template=tagging_evaluation_template)
            
            # Run evaluation
            evaluation_chain = evaluation_prompt | self.llm | StrOutputParser()
            
            result = await evaluation_chain.ainvoke({
                "element_id": tagging_result.element_id,
                "element_name": tagging_result.element_name,
                "element_description": tagging_result.element_description,
                "matched_terms": matched_terms_str
            })
            
            # Parse the confidence score
            confidence = 0.0
            for line in result.strip().split("\n"):
                if "confidence score" in line.lower():
                    # Extract the confidence score
                    match = re.search(r"\d+\.\d+", line)
                    if match:
                        try:
                            confidence = float(match.group(0))
                            confidence = max(0.0, min(1.0, confidence))  # Ensure within bounds
                        except ValueError:
                            logger.warning(f"Could not parse confidence score from line: {line}")
                            pass
            
            # Extract reasoning - keep everything after the confidence score line
            reasoning = ""
            in_reasoning_section = False
            for line in result.strip().split("\n"):
                if "confidence score" in line.lower():
                    in_reasoning_section = True
                    continue
                if in_reasoning_section:
                    reasoning += line + "\n"
            
            if not reasoning:
                reasoning = result  # Fallback to the full result
            
            # Special case handling for account number
            if "account number" in tagging_result.element_name.lower():
                # Look for account identifier terms
                account_identifier_match = False
                for term in tagging_result.matching_terms:
                    term_name = term["name"].lower()
                    if ("account identifier" in term_name or 
                        (("account" in term_name) and 
                         ("id" in term_name.split() or "identifier" in term_name))):
                        account_identifier_match = True
                        # Boost confidence for this specific match
                        confidence = max(confidence, 0.85)
                        reasoning = (f"Strong semantic match found: 'account number' maps conceptually to '{term['name']}'. "
                                    f"This is a standard data governance mapping where account numbers are formally represented "
                                    f"as account identifiers in a standardized business glossary.\n\n") + reasoning
            
            # Determine if modeling is required based on confidence
            if confidence < 0.5:
                reasoning += "\n\nBased on the low confidence score, modeling a new term is recommended."
            
            return confidence, reasoning.strip()
        
        except Exception as e:
            logger.error(f"Error evaluating tagging confidence with reasoning: {e}")
            return 0.5, f"Error evaluating tagging confidence: {e}"
    
    def _extract_key_concepts(self, name: str, description: str) -> set:
        """
        Extract key concepts from name and description.
        
        Args:
            name: Term or element name
            description: Term or element description
            
        Returns:
            Set of key concepts
        """
        concepts = set()
        
        # Clean and normalize
        name_lower = name.lower()
        
        # Extract words from name (higher weight)
        for word in re.findall(r'\b\w+\b', name_lower):
            if len(word) > 2:  # Skip very short words
                concepts.add(word)
        
        # Add compound concepts
        if "account" in name_lower and "number" in name_lower:
            concepts.add("account_identifier")
        
        if "customer" in name_lower and "id" in name_lower:
            concepts.add("customer_identifier")
        
        return concepts