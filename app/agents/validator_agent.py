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
from app.utils.iso_standards import ISO11179Validator # Assuming this utility class exists
from app.utils.cache import cache_manager

logger = logging.getLogger(__name__)

class ValidatorAgent:
    """
    Agent that validates data elements against ISO/IEC 11179 standards,
    focusing on contextual meaning, OPR model for names, and layperson understandability.
    """

    def __init__(self, llm: AzureChatOpenAI):
        """
        Initialize the validator agent.

        Args:
            llm: Language model instance
        """
        self.llm = llm
        # ISO11179Validator can be used for very basic preliminary checks if desired,
        # but the LLM will do the primary contextual and OPR validation.
        self.iso_validator = ISO11179Validator()
        self.approved_acronyms = self._load_approved_acronyms()
        self._setup_validation_chain()

    def _load_approved_acronyms(self):
        """
        Load approved acronyms from CSV file.
        This helps in deciding if an acronym is "universally understood" in context.
        """
        approved_acronyms = {}
        try:
            # Construct path relative to this file's directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up to 'app' then to 'data'
            csv_path = os.path.join(base_dir, "..", "..", "data", "acronyms.csv")
            csv_path = os.path.normpath(csv_path) # Normalize path for OS compatibility

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'acronym' in df.columns and 'definition' in df.columns:
                    for _, row in df.iterrows():
                        # Store acronym as uppercase as they are often written that way
                        approved_acronyms[row['acronym'].strip().upper()] = row['definition'].strip()
                logger.info(f"ValidatorAgent: Loaded {len(approved_acronyms)} approved acronyms from {csv_path}")
            else:
                logger.warning(f"ValidatorAgent: Acronyms file not found at {csv_path}.")
        except Exception as e:
            logger.error(f"ValidatorAgent: Error loading approved acronyms from {csv_path}: {e}")
        return approved_acronyms

    def _format_processes_info(self, data_element: DataElement) -> str:
        """
        Format related business process information for the LLM prompt.
        Provides context about how the data element is used.
        """
        if not data_element.processes:
            return "Related Processes: None provided."
        
        processes = data_element.processes
        process_list = []
        for p_data in processes:
            if isinstance(p_data, Process):
                process_list.append(p_data)
            elif isinstance(p_data, dict):
                try:
                    process_list.append(Process(**p_data))
                except Exception as e:
                    logger.warning(f"ValidatorAgent: Could not convert dict to Process object: {p_data}. Error: {e}")
            else:
                logger.warning(f"ValidatorAgent: Unknown process type encountered: {type(p_data)}")
        
        if not process_list:
            return "Related Processes: None available after formatting."

        info = "Related Processes (for contextual understanding):\n"
        for i, process in enumerate(process_list, 1):
            info += f"  Process {i} Name: {process.process_name}"
            if process.process_id: # Include ID if available
                 info += f" (ID: {process.process_id})"
            if process.process_description:
                # Truncate long descriptions for prompt brevity
                desc_preview = process.process_description[:100] + "..." if len(process.process_description) > 100 else process.process_description
                info += f", Description: {desc_preview}"
            info += "\n"
        return info.strip()

    def _setup_validation_chain(self):
        """
        Set up the LangChain prompt and chain for data element validation.
        The prompt guides the LLM to evaluate based on ISO/IEC 11179 principles,
        contextual sense, OPR model, and layperson understandability.
        """
        template = """
        You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to evaluate the
        given data element's "Current Name" and "Current Description" for quality, contextual sense, and adherence
        to the principles of ISO/IEC 11179, particularly focusing on understandability by a general business audience.

        **Key ISO/IEC 11179 Principles for Validation:**

        **Data Element Names:**
        1.  **Conceptual Structure (Object-Property-Representation - OPR):** A high-quality name should clearly imply:
            * An **Object Class** (e.g., "Customer", "Product", "Sales Order"). This is the main entity.
            * A **Property** (e.g., "Identifier", "Name", "Status", "Effective Date", "Net Amount"). This is a characteristic of the object.
            * Optionally, a **Representation Qualifier** if it adds essential clarity (e.g., "Code", "Text", "Indicator", "Count").
            The name should naturally convey these components in a business-friendly phrase (e.g., "Customer Full Name", "Product Status Code", "Order Total Amount").
        2.  **Clarity & Simplicity for Layperson:** The name MUST be easily understandable by a business user who may not be a domain expert. Avoid internal jargon or overly technical terms.
        3.  **Unambiguity:** The name should have one clear meaning and not be open to multiple interpretations.
        4.  **Conciseness:** Be as brief as possible while ensuring the meaning is fully conveyed.
        5.  **Formatting:**
            * Use consistent, business-readable casing (e.g., "customer name" or "Customer Name" are acceptable; "customerName" or "customer_name" are not).
            * Avoid special characters (e.g., %, &, *, #, /) unless part of a widely recognized term (very rare). Spaces are used for multi-word names.
        6.  **Acronyms/Abbreviations:** Should only be used if they are universally understood (e.g., "ID", "URL", "VAT"). If an acronym is present, assess if it meets this high bar for understandability.

        **Data Element Descriptions:**
        1.  **Clear and Precise Definition:** Must clearly, precisely, and completely define what the data element *is* and what it represents.
        2.  **Contextual Sense & Layperson Readability:** The description MUST make sense in relation to the name and any provided examples or processes. It must be easily readable and understandable by a general business audience.
        3.  **Completeness:** Cover the concept fully.
        4.  **Objectivity:** Be factual and avoid opinions.
        5.  **Grammar & Structure:** Use complete, grammatically correct sentences. Start with a capital letter and end with appropriate punctuation (usually a period).

        **Data Element to Evaluate:**
        - ID: {id}
        - Current Name: {name}
        - Current Description: {description}
        - Example (if provided for context): {example}
        {processes_info}

        **Evaluation Task:**
        Based on ALL the above ISO/IEC 11179 principles, focusing on the OPR structure for names, contextual sense, and layperson understandability for both name and description:
        1.  Assess if the "Current Name" effectively implies an Object, Property, (and optionally Representation) and is clear to a layperson.
        2.  Assess if the "Current Description" is clear, complete, precise, and understandable to a layperson, making contextual sense with the name.
        3.  Check for adherence to formatting and grammatical rules.

        **Output Format:**
        Provide your evaluation *strictly* in the following format, with each item on a new line. Do not include any extra formatting, numbering, or markdown:
        Is name valid: [yes/no - "yes" ONLY if it meets ALL criteria including OPR implication, clarity, simplicity, and layperson understandability]
        Name feedback: [Detailed feedback. If not valid, explain ALL reasons, including issues with OPR structure, clarity, simplicity, or formatting. If valid, confirm its strengths and how it meets the criteria.]
        Is description valid: [yes/no - "yes" ONLY if it meets ALL criteria including clarity, completeness, precision, and layperson understandability]
        Description feedback: [Detailed feedback. If not valid, explain ALL reasons for lack of clarity, completeness, or contextual sense. If valid, confirm its strengths.]
        Overall quality: [GOOD, NEEDS_IMPROVEMENT, or POOR. GOOD only if BOTH name and description are fully valid AND make excellent contextual sense for a layperson.]
        Suggested improvements: [List specific, actionable improvements for name and/or description if the "Overall quality" is not GOOD. Focus on achieving OPR, clarity, and simplicity. If GOOD, state "No improvements needed." Each improvement on a new line, starting with "- ". If multiple lines per improvement, indent subsequent lines.]
        """
        self.validation_prompt = PromptTemplate(
            input_variables=["id", "name", "description", "example", "processes_info"],
            template=template)
        self.validation_chain = self.validation_prompt | self.llm | StrOutputParser()

    def _perform_basic_validation(self, data_element: DataElement) -> Tuple[List[str], bool, bool]:
        """
        Performs very basic, non-LLM checks. The main validation is done by the LLM.
        This can provide hints or context if needed but isn't the primary validation method.
        """
        name_to_check = data_element.existing_name or ""
        desc_to_check = data_element.existing_description or ""
        
        # Using the util.iso_standards.ISO11179Validator for basic syntax.
        # Note: The LLM prompt allows for more flexible casing ("customer name" or "Customer Name")
        # so programmatic checks for strict lowercase might conflict. LLM's contextual evaluation is key.
        name_valid_prog, name_feedback_prog = self.iso_validator.validate_name(name_to_check)
        desc_valid_prog, desc_feedback_prog = self.iso_validator.validate_description(desc_to_check)
        
        basic_issues = []
        if not name_valid_prog:
            # Only add if it's a significant structural issue the LLM might miss,
            # e.g., special characters, not just casing.
            if not re.match(r"^[a-zA-Z0-9 ]+$", name_to_check.replace(" ", "")): # Check for invalid chars
                 basic_issues.append(f"Programmatic Name Check (Syntax): {name_feedback_prog}")
        if not desc_valid_prog:
            # Only add if it's a significant structural issue.
            if not desc_to_check.strip() or not desc_to_check.strip()[-1] in ['.','!','?']:
                basic_issues.append(f"Programmatic Desc Check (Structure): {desc_feedback_prog}")
            
        return basic_issues, name_valid_prog, desc_valid_prog

    def _parse_validation_result(self, result_str: str) -> ValidationResult:
        """
        Parses the LLM's string output into a ValidationResult object.
        Ensures plain text extraction for all fields.
        """
        lines = result_str.strip().split("\n")
        
        is_name_valid_str = ""
        name_feedback_str = ""
        is_desc_valid_str = ""
        desc_feedback_str = ""
        quality_status_str = ""
        improvements_list = []

        # State machine for parsing multi-line feedback/improvements
        current_parsing_target = None # Can be "name_feedback", "desc_feedback", or "improvements"

        for line in lines:
            line_content = line.strip()
            if not line_content: continue # Skip empty lines

            # Check for section headers first
            if line_content.lower().startswith("is name valid:"):
                is_name_valid_str = line_content.split(":", 1)[1].strip().lower()
                current_parsing_target = None
            elif line_content.lower().startswith("name feedback:"):
                name_feedback_str = line_content.split(":", 1)[1].strip()
                current_parsing_target = "name_feedback"
            elif line_content.lower().startswith("is description valid:"):
                is_desc_valid_str = line_content.split(":", 1)[1].strip().lower()
                current_parsing_target = None
            elif line_content.lower().startswith("description feedback:"):
                desc_feedback_str = line_content.split(":", 1)[1].strip()
                current_parsing_target = "desc_feedback"
            elif line_content.lower().startswith("overall quality:"):
                quality_status_str = line_content.split(":", 1)[1].strip().upper()
                current_parsing_target = None
            elif line_content.lower().startswith("suggested improvements:"):
                first_improvement = line_content.split(":", 1)[1].strip()
                # Handle "No improvements needed."
                if first_improvement and first_improvement.lower() != "no improvements needed.":
                    improvements_list.append(first_improvement.lstrip("- "))
                current_parsing_target = "improvements"
            # If it's not a new section header, append to the current target
            elif current_parsing_target == "name_feedback":
                name_feedback_str += " " + line_content
            elif current_parsing_target == "desc_feedback":
                desc_feedback_str += " " + line_content
            elif current_parsing_target == "improvements":
                # Ensure not to add "No improvements needed." if it's a standalone line after header
                if line_content and line_content.lower() != "no improvements needed.":
                    improvements_list.append(line_content.lstrip("- "))
        
        is_name_valid = "yes" in is_name_valid_str
        is_desc_valid = "yes" in is_desc_valid_str

        quality_status = DataQualityStatus.NEEDS_IMPROVEMENT # Default
        if quality_status_str == "GOOD":
            quality_status = DataQualityStatus.GOOD
        elif quality_status_str == "POOR":
            quality_status = DataQualityStatus.POOR
        
        # Consistency check: if LLM says GOOD but marks name/desc as invalid, downgrade.
        if quality_status == DataQualityStatus.GOOD and (not is_name_valid or not is_desc_valid):
            quality_status = DataQualityStatus.NEEDS_IMPROVEMENT
            # Ensure improvements list isn't just "No improvements needed."
            if not improvements_list or (len(improvements_list)==1 and improvements_list[0].lower() == "no improvements needed."):
                 improvements_list = ["Review name and description for full compliance as per detailed feedback."]
        
        # Consolidate feedback strings
        combined_feedback = f"Name Feedback: {name_feedback_str or 'Not provided.'}\nDescription Feedback: {desc_feedback_str or 'Not provided.'}"
        
        # Clean up improvements list if it only contains "No improvements needed."
        if len(improvements_list) == 1 and improvements_list[0].lower() == "no improvements needed.":
            improvements_list = []

        return ValidationResult(
            is_valid=is_name_valid and is_desc_valid, # Overall validity based on LLM's assessment
            quality_status=quality_status,
            feedback=combined_feedback.strip(), # Ensure it's plain text
            suggested_improvements=improvements_list # Ensure it's list of plain strings
        )

    @cache_manager.async_cached(ttl=3600) # Cache LLM validation results for an hour
    async def validate(self, data_element: DataElement) -> ValidationResult:
        """
        Validates a data element using the LLM based on ISO/IEC 11179, OPR, and clarity.
        """
        try:
            processes_info_str = self._format_processes_info(data_element)
            
            # The LLM is expected to do the primary, nuanced validation.
            # Basic programmatic checks can be logged or used as minor context if needed.
            # basic_issues, _, _ = self._perform_basic_validation(data_element)
            # if basic_issues:
            #    logger.debug(f"Basic programmatic checks for {data_element.id} found: {basic_issues}")

            llm_response_str = await self.validation_chain.ainvoke({
                "id": data_element.id,
                "name": data_element.existing_name or "", # Ensure not None
                "description": data_element.existing_description or "", # Ensure not None
                "example": data_element.example or "Not provided.",
                "processes_info": processes_info_str
            })
            
            parsed_result = self._parse_validation_result(llm_response_str)

            # Log if parsing seems to have missed crucial parts, e.g., POOR quality with no improvements
            if parsed_result.quality_status == DataQualityStatus.POOR and not parsed_result.suggested_improvements:
                 logger.warning(f"Validation for {data_element.id} resulted in POOR quality with no improvements. Raw LLM: '{llm_response_str[:200]}...', Parsed: {parsed_result.feedback}")

            return parsed_result

        except Exception as e:
            logger.error(f"Error validating data element {data_element.id}: {e}", exc_info=True)
            # Return a clear error state
            return ValidationResult(
                is_valid=False,
                quality_status=DataQualityStatus.POOR,
                feedback=f"System error during validation: {str(e)}",
                suggested_improvements=["A system error occurred. Please review manually or retry the validation."]
            )

    async def batch_validate(self, data_elements: List[DataElement]) -> List[ValidationResult]:
        """
        Validates multiple data elements in parallel.
        """
        tasks = [self.validate(element) for element in data_elements]
        return await asyncio.gather(*tasks)

