import re
import os
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from app.core.models import DataElement, EnhancementResult, ValidationResult, DataQualityStatus, Process
from app.agents.validator_agent import ValidatorAgent

logger = logging.getLogger(__name__)

class EnhancerAgent:
    def __init__(self, llm: AzureChatOpenAI, acronym_file_path=None):
        self.llm = llm
        self.acronyms = self._load_acronyms(acronym_file_path)
        self._setup_enhancement_chain()
        # Initialize validator agent for quality feedback
        self.validator = ValidatorAgent(llm)
    
    def _load_acronyms(self, acronym_file_path=None):
        """Load acronyms and their definitions from a CSV file."""
        acronyms = {}
        try:
            # Use default path if not provided
            if not acronym_file_path:
                acronym_file_path = "data/acronyms.csv"
            
            # Check if file exists
            if os.path.exists(acronym_file_path):
                df = pd.read_csv(acronym_file_path)
                
                # Ensure required columns exist
                if 'acronym' in df.columns and 'definition' in df.columns:
                    # Group by acronym to handle multiple definitions
                    for acronym, group in df.groupby('acronym'):
                        acronyms[acronym.strip().upper()] = group['definition'].tolist()
                    
                    logger.info(f"Loaded {len(acronyms)} acronyms from {acronym_file_path}")
                else:
                    logger.warning(f"CSV file {acronym_file_path} does not contain required columns 'acronym' and 'definition'")
            else:
                logger.warning(f"Acronym file not found at {acronym_file_path}")
        except Exception as e:
            logger.error(f"Error loading acronyms: {e}")
        
        return acronyms
    
    def _setup_enhancement_chain(self):
        template = """
        You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to enhance the 
        given data element name and description to meet these standards and represent high-quality metadata.
        
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
        
        Data Element to Enhance:
        - ID: {id}
        - Current Name: {name}
        - Current Description: {description}
        - Example (if provided): {example}
        {processes_info}
        
        Additional Context:
        {additional_context}
        
        Validation Feedback:
        {validation_feedback}
        
        Acronym Information:
        {acronym_info}
        
        Here are examples of high-quality data elements that meet ISO/IEC 11179 standards:
        
        Example 1:
        - Original Name: cust_id
        - Enhanced Name: customer identifier
        - Original Description: Customer ID in the system
        - Enhanced Description: A unique alphanumeric code that identifies an individual customer within the organization's systems.
        - Quality Assessment: GOOD
        
        Example 2:
        - Original Name: LN
        - Enhanced Name: last name
        - Original Description: Last name
        - Enhanced Description: The legal surname of an individual as it appears on official identification documents.
        - Quality Assessment: GOOD
        
        Example 3:
        - Original Name: trans_amt
        - Enhanced Name: transaction amount
        - Original Description: The $ amnt of the trans.
        - Enhanced Description: The monetary value of a transaction expressed in the transaction's currency.
        - Quality Assessment: GOOD
        
        Example 4:
        - Original Name: DOB
        - Enhanced Name: date of birth
        - Original Description: DOB of customer
        - Enhanced Description: The calendar date on which an individual person was born, typically stored in a standardized date format.
        - Quality Assessment: GOOD
        
        Example 5:
        - Original Name: acct_bal
        - Enhanced Name: account balance
        - Original Description: Current bal. in the account
        - Enhanced Description: The current monetary value held in an account, calculated as the sum of all deposits minus all withdrawals and fees.
        - Quality Assessment: GOOD
        
        Based on the ISO/IEC 11179 standards, the validation feedback, the acronym information, and the examples, enhance this data element.
        
        - Expand acronyms and abbreviations according to the provided acronym definitions, choosing the most contextually appropriate expansion.
        - If acronyms are encountered that aren't in the provided list, expand them if you know their meaning.
        - Only keep very well-known acronyms (like PDF, URL, ID) that are universally understood.
        - For descriptions, ensure they are complete sentences with proper grammar and explain the concept fully.
        - Make sure the enhanced element would receive a "GOOD" quality assessment.
        
        Provide your enhancement as follows:
        1. Enhanced Name: [provide the improved name - MUST be in lowercase with spaces between words]
        2. Enhanced Description: [provide the improved description]
        3. Enhancement Notes: [explain the changes made and how they improve the compliance with standards]
        4. Confidence Score (0.0-1.0): [provide a confidence score for the enhancement]
        
        IMPORTANT: Do not use any special formatting like asterisks (**) or other markdown formatting in your response.
        
        Make the enhancement business-friendly and ensure they make sense in a business context.
        The names should be simple enough for business users to understand without being technical.
        Use the example and process information provided to add context to your enhancements.
        """
        
        self.enhancement_prompt = PromptTemplate(
            input_variables=[
                "id", "name", "description", "example", "processes_info", 
                "validation_feedback", "acronym_info", "additional_context"
            ],
            template=template)
        self.enhancement_chain = self.enhancement_prompt | self.llm | StrOutputParser()
    
    def _prepare_acronym_info(self, data_element: DataElement) -> str:
        """
        Prepare acronym information for the prompt based on detected acronyms in the data element.
        
        Args:
            data_element: The data element to analyze for acronyms
            
        Returns:
            str: Formatted acronym information
        """
        # Extract potential acronyms from name and description
        name = data_element.existing_name
        description = data_element.existing_description
        
        # Simple regex to find uppercase words that might be acronyms (2+ characters)
        potential_acronyms = set(re.findall(r'\b[A-Z]{2,}\b', name + " " + description))
        
        if not potential_acronyms and not self.acronyms:
            return "No acronyms detected or provided."
        
        # Prepare information about found acronyms
        acronym_info = "Known acronyms in this context:\n"
        
        found_in_element = False
        for acronym in potential_acronyms:
            if acronym in self.acronyms:
                definitions = self.acronyms[acronym]
                found_in_element = True
                if len(definitions) == 1:
                    acronym_info += f"- {acronym}: {definitions[0]}\n"
                else:
                    acronym_info += f"- {acronym}: Multiple definitions available:\n"
                    for i, definition in enumerate(definitions, 1):
                        acronym_info += f"  {i}. {definition}\n"
        
        # Add a sample of other acronyms from our dictionary that might be relevant
        # (limit to 10 to avoid overwhelming the prompt)
        other_acronyms = [acronym for acronym in self.acronyms if acronym not in potential_acronyms]
        if other_acronyms:
            sample_size = min(10, len(other_acronyms))
            sample_acronyms = other_acronyms[:sample_size]
            
            acronym_info += "\nSample of other known acronyms in the domain:\n"
            for acronym in sample_acronyms:
                definitions = self.acronyms[acronym]
                if len(definitions) == 1:
                    acronym_info += f"- {acronym}: {definitions[0]}\n"
                else:
                    acronym_info += f"- {acronym}: Multiple definitions available\n"
            
            if len(other_acronyms) > sample_size:
                acronym_info += f"(Plus {len(other_acronyms) - sample_size} more acronyms in the database)\n"
        elif not found_in_element:
            return "No known acronyms detected in this data element."
        
        acronym_info += "\nInstructions for acronym handling:\n"
        acronym_info += "- Expand acronyms in both name and description unless they are universally understood (like URL, PDF)\n"
        acronym_info += "- Choose the most contextually appropriate expansion when multiple definitions are available\n"
        acronym_info += "- If you know the meaning of acronyms not listed above, expand those as well\n"
        
        return acronym_info
    
    def _format_processes_info(self, data_element: DataElement) -> str:
        """
        Format the processes information for the prompt.
        
        Args:
            data_element: The data element containing processes
            
        Returns:
            str: Formatted processes information
        """
        if not data_element.processes:
            return "- Related Processes: None"
        
        processes = data_element.processes
        # Ensure processes are Process objects
        if not all(isinstance(p, Process) for p in processes):
            # Try to convert dictionaries to Process objects
            processes = []
            for p in data_element.processes:
                if isinstance(p, Process):
                    processes.append(p)
                elif isinstance(p, dict):
                    processes.append(Process(**p))
                else:
                    logger.warning(f"Unknown process type: {type(p)}")
        
        processes_info = "- Related Processes:\n"
        
        for i, process in enumerate(processes, 1):
            processes_info += f"  Process {i} ID: {process.process_id}\n"
            processes_info += f"  Process {i} Name: {process.process_name}\n"
            if process.process_description:
                processes_info += f"  Process {i} Description: {process.process_description}\n"
            processes_info += "\n"
        
        return processes_info
    
    def _parse_enhancement_result(self, result: str)->EnhancementResult:
        enhanced_name = ""
        enhanced_description = ""
        feedback = ""
        confidence = 0.7  # Default confidence
        
        # Clean up result by removing any markdown or special characters
        result = result.replace("**", "").strip()
        
        lines = result.strip().split("\n")
        for line in lines:
            if "Enhanced Name:" in line:
                enhanced_name = line.split("Enhanced Name:")[1].strip()
                # Remove any quotes if present
                enhanced_name = enhanced_name.strip('"\'[]')
                break
        
        description_start = None
        description_end = None
        
        for i, line in enumerate(lines):
            if "Enhanced Description:" in line:
                description_start = i
            elif description_start is not None and "Enhancement Notes:" in line:
                description_end = i
                break
        
        if description_start is not None:
            if description_end is not None:
                description_lines = lines[description_start:description_end]
            else:
                description_lines = lines[description_start:]
            
            for i, line in enumerate(description_lines):
                if "Enhanced Description:" in line:
                    enhanced_description = line.split("Enhanced Description:")[1].strip()
                    enhanced_description += " " + " ".join([l.strip() for l in description_lines[i+1:]])
            
            # Remove any quotes or brackets if present
            enhanced_description = enhanced_description.strip('"\'[]')
        
        notes_start = None
        notes_end = None
        for i, line in enumerate(lines):
            if "Enhancement Notes:" in line:
                notes_start = i
            elif notes_start is not None and "Confidence Score" in line:
                notes_end = i
                break
        
        if notes_start is not None:
            if notes_end is not None:
                feedback_lines = lines[notes_start:notes_end]
            else:
                feedback_lines = lines[notes_start:]
            
            for i, line in enumerate(feedback_lines):
                if "Enhancement Notes:" in line:
                    feedback = line.split("Enhancement Notes:")[1].strip()
                    feedback += " " + " ".join([l.strip() for l in feedback_lines[i+1:]])
        
        for line in lines:
            if "Confidence Score" in line:
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    try:
                        confidence = float(match.group(1))
                    except ValueError:
                        logger.warning(f"Failed to parse confidence score from: {line}")
                        pass
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return EnhancementResult(
            enhanced_name=enhanced_name,
            enhanced_description=enhanced_description,
            feedback=feedback,
            confidence=confidence
        )
    
    async def validate_enhanced_element(self, data_element: DataElement) -> ValidationResult:
        """
        Validate an enhanced data element to check its quality.
        
        Args:
            data_element: The enhanced data element to validate
            
        Returns:
            ValidationResult: Validation feedback and quality assessment
        """
        # Use validator agent to check quality
        return await self.validator.validate(data_element)
    
    async def enhance_until_quality(self, data_element: DataElement, 
                                    validation_feedback: str = "", 
                                    additional_context: str = "",
                                    max_iterations: int = 3) -> Tuple[EnhancementResult, ValidationResult]:
        """
        Enhance a data element repeatedly until quality is good or max iterations reached.
        
        Args:
            data_element: The data element to enhance
            validation_feedback: Initial validation feedback
            additional_context: Additional context to help enhancement
            max_iterations: Maximum number of enhancement iterations
            
        Returns:
            Tuple[EnhancementResult, ValidationResult]: The final enhanced result and validation
        """
        current_element = data_element
        current_feedback = validation_feedback
        iteration = 0
        last_enhancement = None
        last_validation = None
        
        # Enhancement loop
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Enhancement iteration {iteration} for element: {current_element.id}")
            
            # Get current enhancement
            enhancement_result = await self.enhance(current_element, current_feedback, additional_context)
            
            # Create a test element with the enhanced data
            test_element = DataElement(
                id=current_element.id,
                existing_name=enhancement_result.enhanced_name,
                existing_description=enhancement_result.enhanced_description,
                example=current_element.example,
                processes=current_element.processes,
                cdm=current_element.cdm
            )
            
            # Validate the enhanced element
            validation_result = await self.validate_enhanced_element(test_element)
            
            # Save the current state
            last_enhancement = enhancement_result
            last_validation = validation_result
            
            # If quality is good, we're done
            if validation_result.quality_status == DataQualityStatus.GOOD:
                logger.info(f"Enhancement reached GOOD quality after {iteration} iterations")
                break
                
            # Otherwise, prepare for next iteration
            # Use the enhanced element as the new input
            current_element = test_element
            
            # Set feedback for next iteration
            current_feedback = validation_result.feedback
            if validation_result.suggested_improvements:
                current_feedback += "\n\nSuggested improvements:\n" + "\n".join(
                    f"- {improvement}" for improvement in validation_result.suggested_improvements
                )
        
        if iteration >= max_iterations and last_validation.quality_status != DataQualityStatus.GOOD:
            logger.warning(f"Reached maximum iterations ({max_iterations}) without achieving GOOD quality")
            
        return last_enhancement, last_validation
    
    async def enhance(self, data_element: DataElement, validation_feedback: str = "", additional_context: str = "") -> EnhancementResult:
        """
        Enhance a data element based on validation feedback and additional context.
        
        Args:
            data_element: The data element to enhance
            validation_feedback: Feedback from validation process
            additional_context: Additional contextual information to help enhancement
            
        Returns:
            EnhancementResult: The enhanced data element
        """
        try:
            # Prepare acronym information
            acronym_info = self._prepare_acronym_info(data_element)
            
            # Format processes information
            processes_info = self._format_processes_info(data_element)
            
            result = await self.enhancement_chain.ainvoke({
                "id": data_element.id,
                "name": data_element.existing_name,
                "description": data_element.existing_description,
                "example": data_element.example or "Not provided",
                "processes_info": processes_info,
                "validation_feedback": validation_feedback,
                "acronym_info": acronym_info,
                "additional_context": additional_context
            })
            
            enhancement_result = self._parse_enhancement_result(result)
            
            # Log enhancement result for debugging
            logger.info(f"Enhanced name: {enhancement_result.enhanced_name}")
            logger.info(f"Enhanced description: {enhancement_result.enhanced_description}")
            logger.info(f"Confidence: {enhancement_result.confidence}")
            
            return enhancement_result
        except Exception as e:
            logger.error(f"Error enhancing data element: {e}")
            # Return minimal enhancement in case of error
            return EnhancementResult(
                enhanced_name=data_element.existing_name,
                enhanced_description=data_element.existing_description,
                feedback=f"Error during enhancement: {str(e)}",
                confidence=0.0
            )