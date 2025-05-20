"""
Validator Agent - Validates data elements against ISO/IEC 11179 standards.
"""

from typing import Dict, Any, List, Tuple, Optional
import re
import logging
import os
import pandas as pd
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from app.core.models import DataElement, ValidationResult, DataQualityStatus, Process
from app.utils.iso_standards import ISO11179Validator
from app.utils.cache import cache_manager

logger = logging.getLogger(__name__)

class ValidatorAgent:
    """
    Agent that validates data elements against ISO/IEC 11179 standards.
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        """
        Initialize the validator agent.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
        self.iso_validator = ISO11179Validator()
        self.approved_acronyms = self._load_approved_acronyms()
        self._setup_validation_chain()
    
    def _load_approved_acronyms(self):
        """
        Load approved acronyms from CSV file.
        
        Returns:
            Dict mapping acronyms to definitions
        """
        approved_acronyms = {}
        try:
            csv_path = os.path.join("data", "acronyms.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'acronym' in df.columns and 'definition' in df.columns:
                    for _, row in df.iterrows():
                        approved_acronyms[row['acronym'].strip().upper()] = row['definition'].strip()
                logger.info(f"Loaded {len(approved_acronyms)} approved acronyms from {csv_path}")
            else:
                logger.warning(f"Acronyms file not found at {csv_path}")
        except Exception as e:
            logger.error(f"Error loading approved acronyms: {e}")
        
        return approved_acronyms
    
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
        
        processes_info = "- Related Processes:\n"
        
        for i, process in enumerate(data_element.processes, 1):
            processes_info += f"  Process {i} ID: {process.process_id}\n"
            processes_info += f"  Process {i} Name: {process.process_name}\n"
            if process.process_description:
                processes_info += f"  Process {i} Description: {process.process_description}\n"
            processes_info += "\n"
        
        return processes_info
    
    def _setup_validation_chain(self):
        """Set up the LangChain chain for validation."""
        template = """
        You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to evaluate the 
        given data element name and description against these standards to determine the quality of the metadata.
        
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
        
        Data Element to Evaluate:
        - ID: {id}
        - Current Name: {name}
        - Current Description: {description}
        - Example (if provided): {example}
        {processes_info}
        
        Here are examples of high-quality data elements that meet ISO/IEC 11179 standards:
        
        Example 1:
        - Original Name: cust_id
        - Enhanced Name: customer identifier
        - Original Description: Customer ID in the system
        - Enhanced Description: A unique alphanumeric code that identifies an individual customer within the organization's systems.
        - Quality Assessment: GOOD
        - Reason: The name is in lowercase with spaces, avoids technical terms, and is clear. The description is complete, precise, and explains exactly what the element represents.
        
        Example 2:
        - Original Name: LN
        - Enhanced Name: last name
        - Original Description: Last name
        - Enhanced Description: The legal surname of an individual as it appears on official identification documents.
        - Quality Assessment: GOOD
        - Reason: The name is properly expanded and formatted. The description clearly defines what the element represents with sufficient context.
        
        Example 3:
        - Original Name: trans_amt
        - Enhanced Name: transaction amount
        - Original Description: The $ amnt of the trans.
        - Enhanced Description: The monetary value of a transaction expressed in the transaction's currency.
        - Quality Assessment: GOOD
        - Reason: The abbreviated name is properly expanded and formatted. The description provides clear context and fixes abbreviations and grammar issues.
        
        Based on the ISO/IEC 11179 standards, evaluate the quality of this data element.
        
        IMPORTANT: Your response MUST include ALL of the following 6 evaluation points in order:
        1. Is the name valid according to ISO/IEC 11179 standards? [yes/no]
        2. Detailed feedback on the name quality (provide specific issues and suggestions)
        3. Is the description valid according to the standards? [yes/no]
        4. Detailed feedback on the description quality (pay careful attention to grammar and punctuation)
        5. Overall quality status - MUST be one of: "GOOD", "NEEDS_IMPROVEMENT", or "POOR"
        6. List of specific improvements that could be made (be detailed and prescriptive)
        
        DO NOT use any special formatting characters like asterisks (**), backticks (``), or markdown syntax.
        DO NOT include any \\n characters or other special characters in your response.
        Please be thorough and specific in your feedback, as it will be used to improve the data element.
        """
        
        self.validation_prompt = PromptTemplate(
            input_variables=["id", "name", "description", "example", "processes_info"],
            template=template)
        self.validation_chain = self.validation_prompt | self.llm | StrOutputParser()
    
    def _perform_basic_validation(self, data_element: DataElement) -> Tuple[List[str], bool, bool]:
        """
        Perform basic validation checks without using the LLM.
        
        Args:
            data_element: The data element to validate
            
        Returns:
            Tuple containing:
            - List of validation issues
            - Whether the name is valid
            - Whether the description is valid
        """
        basic_validation_issues = []
        
        # Validate name
        name_valid, name_feedback = self.iso_validator.validate_name(data_element.existing_name)
        if not name_valid:
            basic_validation_issues.append(f"Name validation failed: {name_feedback}")
        
        # Validate description
        desc_valid, desc_feedback = self.iso_validator.validate_description(data_element.existing_description)
        if not desc_valid:
            basic_validation_issues.append(f"Description validation failed: {desc_feedback}")
        
        # Check for acronyms (max 3 characters)
        words = re.findall(r'\b[A-Z]{2,3}\b', data_element.existing_name + " " + data_element.existing_description)
        acronym_info = ""
        
        if words:
            approved_acronyms_found = []
            contextual_acronyms = []
            
            for word in words:
                if word in self.approved_acronyms:
                    approved_acronyms_found.append(f"{word} ({self.approved_acronyms[word]})")
                else:
                    contextual_acronyms.append(word)
            
            # Add acronym info to feedback but don't mark invalid
            if approved_acronyms_found:
                acronym_info += f"\nFound approved acronyms: {', '.join(approved_acronyms_found)}"
            if contextual_acronyms:
                acronym_info += f"\nFound acronyms without context: {', '.join(contextual_acronyms)}"
            
            if acronym_info:
                basic_validation_issues.append(acronym_info)
        
        return basic_validation_issues, name_valid, desc_valid
    
    def _parse_validation_result(self, result: str) -> ValidationResult:
        """
        Parse the validation result from the LLM.
        
        Args:
            result: The LLM output
            
        Returns:
            ValidationResult: Parsed validation result
        """
        # Clean up the result
        result = result.replace("**", "").replace("```", "")
        lines = result.strip().split("\n")
        
        is_name_valid = False
        is_desc_valid = False
        name_feedback = ""
        desc_feedback = ""
        quality_status = DataQualityStatus.NEEDS_IMPROVEMENT  # Default to NEEDS_IMPROVEMENT instead of POOR
        improvements = []
        
        # Extract each section
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("1."):
                current_section = "name_valid"
                if "yes" in line.lower():
                    is_name_valid = True
            elif line.startswith("2."):
                current_section = "name_feedback"
                name_feedback = line[2:].strip()
            elif line.startswith("3."):
                current_section = "desc_valid"
                if "yes" in line.lower():
                    is_desc_valid = True
            elif line.startswith("4."):
                current_section = "desc_feedback"
                desc_feedback = line[2:].strip()
            elif line.startswith("5."):
                current_section = "quality_status"
                if "GOOD" in line:
                    quality_status = DataQualityStatus.GOOD
                elif "NEEDS_IMPROVEMENT" in line or "NEEDS IMPROVEMENT" in line:
                    quality_status = DataQualityStatus.NEEDS_IMPROVEMENT
                elif "POOR" in line:
                    quality_status = DataQualityStatus.POOR
            elif line.startswith("6."):
                current_section = "improvements"
            elif current_section == "name_feedback":
                name_feedback += " " + line
            elif current_section == "desc_feedback":
                desc_feedback += " " + line
            elif current_section == "improvements":
                if line.startswith("-") or line.startswith("*") or (line[0].isdigit() and ". " in line):
                    # This is a new improvement item
                    line = re.sub(r'^[-*]\s+|\d+\.\s+', '', line)
                    improvements.append(line)
                elif improvements:
                    # This is a continuation of the previous improvement
                    improvements[-1] += " " + line
                else:
                    # If no previous improvement item to append to, create a new one
                    improvements.append(line)
        
        # Ensure quality status is consistent with validations
        if is_name_valid and is_desc_valid:
            # If both are valid but status wasn't explicitly GOOD, use NEEDS_IMPROVEMENT
            if quality_status == DataQualityStatus.POOR:
                quality_status = DataQualityStatus.NEEDS_IMPROVEMENT
        elif not is_name_valid and not is_desc_valid:
            # If both are invalid, it should be POOR
            quality_status = DataQualityStatus.POOR
        
        # Combine feedback
        combined_feedback = f"Name feedback: {name_feedback}\n\nDescription feedback: {desc_feedback}"
        
        # If no improvements were extracted, create some based on validation results
        if not improvements:
            if not is_name_valid:
                improvements.append("Improve the name to comply with ISO/IEC 11179 standards")
            if not is_desc_valid:
                improvements.append("Enhance the description to be more precise and complete")
            if is_name_valid and is_desc_valid and quality_status != DataQualityStatus.GOOD:
                improvements.append("Further refine name and description for better clarity")
        
        return ValidationResult(
            is_valid=is_name_valid and is_desc_valid,
            quality_status=quality_status, 
            feedback=combined_feedback,
            suggested_improvements=improvements
        )
    
    @cache_manager.async_cached(ttl=3600)  # Cache for 1 hour
    async def validate(self, data_element: DataElement) -> ValidationResult:
        """
        Validate a data element against ISO/IEC 11179 standards with approved acronyms.
        
        Args:
            data_element: The data element to validate
            
        Returns:
            ValidationResult: The validation result
        """
        try:
            # First, perform basic validation
            basic_validation_issues, name_valid, desc_valid = self._perform_basic_validation(data_element)
            
            # Format processes information
            processes_info = self._format_processes_info(data_element)
            
            # Build context information
            context_info = ""
            if basic_validation_issues:
                context_info += "Basic validation issues detected:\n"
                for issue in basic_validation_issues:
                    context_info += f"- {issue}\n"
                context_info += "\nPlease validate both name and description comprehensively regardless.\n\n"
            
            # Add the context to the template if we have any
            template = self.validation_prompt.template
            if context_info:
                template = template.replace("Data Element to Evaluate:", 
                                         f"Additional Context Information:\n{context_info}\nData Element to Evaluate:")
            
            # Create a custom prompt for this validation
            custom_prompt = PromptTemplate(
                input_variables=["id", "name", "description", "example", "processes_info"],
                template=template)
            custom_chain = custom_prompt | self.llm | StrOutputParser()
            
            # Perform detailed evaluation
            result = await custom_chain.ainvoke({
                "id": data_element.id,
                "name": data_element.existing_name,
                "description": data_element.existing_description,
                "example": data_element.example or "Not provided",
                "processes_info": processes_info
            })
            
            return self._parse_validation_result(result)
        except Exception as e:
            logger.error(f"Error validating data element: {e}")
            # Return a minimal validation result in case of error
            return ValidationResult(
                is_valid=False,
                quality_status=DataQualityStatus.POOR,
                feedback=f"Error during validation: {str(e)}",
                suggested_improvements=["Retry validation after resolving the error"]
            )
    
    async def batch_validate(self, data_elements: List[DataElement]) -> List[ValidationResult]:
        """
        Validate multiple data elements in parallel.
        
        Args:
            data_elements: List of data elements to validate
            
        Returns:
            List of validation results
        """
        # Create tasks for all validations
        tasks = [self.validate(element) for element in data_elements]
        
        # Run all tasks concurrently
        return await asyncio.gather(*tasks)