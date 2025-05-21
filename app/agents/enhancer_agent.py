"""
Enhancer Agent - Enhances data elements to meet ISO/IEC 11179 standards.
Focuses on contextual sense, OPR model, layperson understandability, and enhances only if needed.
"""

import re
import os
import logging
import pandas as pd
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from app.core.models import DataElement, EnhancementResult, ValidationResult, DataQualityStatus, Process
from app.agents.validator_agent import ValidatorAgent # To get validation feedback
from app.utils.cache import cache_manager

logger = logging.getLogger(__name__)

class EnhancerAgent:
    """
    Agent that enhances data elements to meet ISO/IEC 11179 standards,
    prioritizing contextual meaning, OPR model for names, layperson understandability,
    and the principle of "enhance only if needed".
    """

    def __init__(self, llm: AzureChatOpenAI, acronym_file_path=None):
        """
        Initialize the enhancer agent.

        Args:
            llm: Language model instance
            acronym_file_path: Path to acronym definitions file (optional)
        """
        self.llm = llm
        self.acronyms = self._load_acronyms(acronym_file_path)
        self._setup_enhancement_chain()
        # Validator agent is used to assess quality of enhanced outputs if needed by the workflow
        self.validator = ValidatorAgent(llm)

    def _load_acronyms(self, acronym_file_path=None):
        """
        Load acronyms and their definitions from a CSV file.
        Used to guide the LLM on expanding acronyms appropriately.
        """
        acronyms = {}
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_dir, "..", "..", "data", "acronyms.csv")
            actual_path = acronym_file_path if acronym_file_path else default_path
            actual_path = os.path.normpath(actual_path)

            if os.path.exists(actual_path):
                df = pd.read_csv(actual_path)
                if 'acronym' in df.columns and 'definition' in df.columns:
                    for _, row in df.iterrows():
                        # Store one primary definition, assuming CSV is curated for this
                        acronyms[row['acronym'].strip().upper()] = row['definition'].strip()
                logger.info(f"EnhancerAgent: Loaded {len(acronyms)} acronym definitions from {actual_path}")
            else:
                logger.warning(f"EnhancerAgent: Acronyms file not found at {actual_path}.")
        except Exception as e:
            logger.error(f"EnhancerAgent: Error loading acronyms from {actual_path}: {e}")
        return acronyms

    def _setup_enhancement_chain(self):
        """
        Set up the LangChain prompt and chain for data element enhancement.
        The prompt emphasizes OPR, clarity, layperson understandability, and "enhance only if needed".
        """
        template = """
        You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to enhance the
        given data element's "Current Name" and "Current Description" to meet these standards.
        The primary goals are:
        1.  **Contextual Sense & Layperson Understandability:** The name and description must be crystal clear to a general business user.
        2.  **Object-Property-Representation (OPR) Model for Names:** Names should clearly imply an Object Class, a Property, and optionally a Representation Qualifier.
        3.  **Enhance Only If Needed:** If the current name and description already meet these high standards, do not change them.

        **Key ISO/IEC 11179 Principles for Enhancement:**

        **Data Element Names:**
        * **OPR Structure:** Clearly identify an **Object Class** (e.g., "Customer", "Product"), a **Property** (e.g., "Identifier", "Name", "Status"), and optionally a **Representation Qualifier** (e.g., "Code", "Text"). Formulate a business-friendly name reflecting this (e.g., "Customer Full Name", "Product Status Code").
        * **Clarity & Simplicity:** Must be easily understood by a non-domain expert. Use simple, direct language.
        * **Formatting:** Prefer consistent, readable casing (e.g., "customer full name" or "Customer Full Name"). Avoid camelCase/snake_case. Use spaces for multi-word names. No special characters (except spaces).
        * **Acronyms:** Expand unless universally known (e.g., ID, URL). Use "Acronym Information" below if provided.

        **Data Element Descriptions:**
        * **Clear, Precise, Complete Definition:** Define what the data element *is*.
        * **Layperson Readability & Context:** Must make sense with the name and be easily understood. Use full sentences, proper grammar, start with a capital, end with a period.
        * **Objectivity:** Factual and unambiguous.

        **Data Element to Potentially Enhance:**
        - ID: {id}
        - Current Name: {name}
        - Current Description: {description}
        - Example (for context): {example}
        {processes_info}

        **Validation Feedback on Current Element (consider this carefully):**
        {validation_feedback}

        **Acronym Information (use to expand acronyms in current name/description if enhancement is needed):**
        {acronym_info}

        **CRITICAL INSTRUCTION: "Enhance Only If Needed"**
        First, thoroughly evaluate if the "Current Name" and "Current Description" ALREADY meet ALL the above standards (OPR structure, clarity for layperson, formatting, description quality).
        * IF YES (they are already excellent and compliant):
            * "Enhanced Name" MUST be identical to "Current Name".
            * "Enhanced Description" MUST be identical to "Current Description".
            * For "Enhancement Notes", state: "Original name and description meet quality standards and are contextually sound; no changes were necessary."
            * "Confidence Score" should be high (e.g., 0.95 or 1.0).
        * IF NO (enhancement is needed to meet the standards):
            * Create an "Enhanced Name" that fully adheres to the OPR model and other name standards.
            * Create an "Enhanced Description" that is clear, complete, and adheres to all description standards.
            * Expand acronyms appropriately.
            * Explain the changes in "Enhancement Notes".
            * Provide an appropriate "Confidence Score".

        **Output Format:**
        Provide your response *strictly* in the following format, with each item on a new line. Do not include any extra formatting, numbering, or markdown:
        Enhanced Name: [Provide the enhanced name as plain text here, or the original if no changes were needed]
        Enhanced Description: [Provide the enhanced description as plain text here, or the original if no changes. Must start with a capital and end with a period.]
        Enhancement Notes: [Explain the changes made (what, why, how it improves) OR specifically state why no changes were made if original was compliant.]
        Confidence Score (0.0-1.0): [Provide a numerical confidence score for your enhancement or assessment of original quality.]
        """
        self.enhancement_prompt = PromptTemplate(
            input_variables=[
                "id", "name", "description", "example", "processes_info",
                "validation_feedback", "acronym_info"
            ],
            template=template)
        self.enhancement_chain = self.enhancement_prompt | self.llm | StrOutputParser()

    def _prepare_acronym_info(self, data_element: DataElement) -> str:
        """
        Prepares a string with relevant acronym expansions for the LLM prompt.
        """
        name = data_element.existing_name or ""
        desc = data_element.existing_description or ""
        # Regex to find potential acronyms: 2-5 uppercase letters, possibly followed by 's' or 'S'
        # and ensuring they are whole words (bounded by non-alphanumeric or start/end of string)
        potential_acronyms_in_text = set(re.findall(r'\b[A-Z][A-Z0-9]{1,4}(?:[sS])?\b', name + " " + desc))
        
        # Filter out universally accepted acronyms that don't need expansion
        common_exceptions = {"ID", "URL", "SKU", "API", "KPI", "VAT", "ETA", "PDF", "SQL"}
        acronyms_to_check = potential_acronyms_in_text - common_exceptions

        if not self.acronyms and not acronyms_to_check:
            return "No specific acronym information. Expand any non-universal acronyms based on common business understanding."

        info_parts = []
        if acronyms_to_check:
            info_parts.append("Consider expanding the following detected potential acronyms if they are not universally understood in this context:")
            for acronym_in_text in sorted(list(acronyms_to_check)): # Sort for consistent prompt
                if acronym_in_text in self.acronyms:
                    info_parts.append(f"- '{acronym_in_text}' (Known expansion: '{self.acronyms[acronym_in_text]}')")
                else:
                    info_parts.append(f"- '{acronym_in_text}' (Definition not in provided list; expand if meaning is not obvious).")
        
        if not info_parts: # If all detected acronyms were common exceptions
            return "Detected acronyms appear to be universally understood. Ensure any other less common acronyms are expanded."

        return "Relevant Acronym Information to Guide Expansion:\n" + "\n".join(info_parts) + "\nGeneral Rule: Expand all acronyms for clarity unless they are as common as 'ID' or 'URL'."

    def _format_processes_info(self, data_element: DataElement) -> str:
        """
        Formats related business process information for the LLM prompt.
        """
        if not data_element.processes:
            return "Related Processes: None provided."
        
        processes = data_element.processes
        process_list = []
        for p_data in processes:
            if isinstance(p_data, Process): process_list.append(p_data)
            elif isinstance(p_data, dict):
                try: process_list.append(Process(**p_data))
                except Exception as e: logger.warning(f"EnhancerAgent: Could not convert dict to Process: {p_data}. Error: {e}")
            else: logger.warning(f"EnhancerAgent: Unknown process type: {type(p_data)}")
        
        if not process_list: return "Related Processes: None available after formatting."

        info = "Related Processes (for contextual understanding of the data element's use):\n"
        for i, process in enumerate(process_list, 1):
            info += f"  Process {i} Name: {process.process_name}"
            if process.process_id: info += f" (ID: {process.process_id})"
            if process.process_description:
                desc_preview = process.process_description[:100] + "..." if len(process.process_description) > 100 else process.process_description
                info += f", Description: {desc_preview}"
            info += "\n"
        return info.strip()

    def _parse_enhancement_result(self, result_str: str) -> EnhancementResult:
        """
        Parses the LLM's string output into an EnhancementResult object.
        Ensures plain text extraction and applies basic formatting rules to description.
        """
        enhanced_name = ""
        enhanced_description = ""
        feedback_notes = "" # For "Enhancement Notes"
        confidence_score_str = "0.5" # Default confidence
        
        current_parsing_target = None

        for line in result_str.strip().split("\n"):
            line_content = line.strip()
            if not line_content: continue

            if line_content.startswith("Enhanced Name:"):
                enhanced_name = line_content.replace("Enhanced Name:", "").strip()
                # Clean potential list markers or other LLM artifacts
                enhanced_name = re.sub(r"^\s*[\d\W.-]*\s*", "", enhanced_name).strip()
                current_parsing_target = None
            elif line_content.startswith("Enhanced Description:"):
                enhanced_description = line_content.replace("Enhanced Description:", "").strip()
                enhanced_description = re.sub(r"^\s*[\d\W.-]*\s*", "", enhanced_description).strip()
                current_parsing_target = "description"
            elif line_content.startswith("Enhancement Notes:"):
                feedback_notes = line_content.replace("Enhancement Notes:", "").strip()
                current_parsing_target = "notes"
            elif line_content.startswith("Confidence Score:"):
                confidence_score_str = line_content.replace("Confidence Score:", "").strip()
                current_parsing_target = None
            elif current_parsing_target == "description":
                enhanced_description += " " + line_content
            elif current_parsing_target == "notes":
                feedback_notes += " " + line_content
        
        # Final cleaning and formatting for description
        if enhanced_description:
            enhanced_description = enhanced_description.strip()
            if enhanced_description: # Ensure not empty after strip
                enhanced_description = enhanced_description[0].upper() + enhanced_description[1:]
                if not enhanced_description.endswith(('.', '!', '?')):
                    enhanced_description += "."
        
        confidence = 0.5 # Default
        try:
            # Try to find a float like "0.9" or "1.0"
            confidence_match = re.search(r"(\b\d\.\d+\b|\b[01]\b)", confidence_score_str) # Matches 0.X, 1.0, 0, 1
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence)) # Clamp to 0.0-1.0
            else:
                logger.warning(f"EnhancerAgent: Could not parse float from confidence score string: '{confidence_score_str}'")
        except ValueError:
            logger.warning(f"EnhancerAgent: ValueError parsing confidence score from: '{confidence_score_str}'")

        # Fallbacks if parsing completely failed for critical fields
        if not enhanced_name:
            logger.warning("EnhancerAgent: Parsing failed to extract Enhanced Name.")
            enhanced_name = "parsing_error_name" # Placeholder to indicate error
        if not enhanced_description:
            logger.warning("EnhancerAgent: Parsing failed to extract Enhanced Description.")
            enhanced_description = "Parsing error: Enhanced description could not be extracted." # Placeholder
            
        return EnhancementResult(
            enhanced_name=enhanced_name,
            enhanced_description=enhanced_description,
            feedback=feedback_notes.strip(), # This is "Enhancement Notes"
            confidence=confidence
        )

    async def validate_enhanced_element(self, data_element_for_validation: DataElement) -> ValidationResult:
        """
        Validates a data element (typically one that has just been enhanced).
        The DataElement passed here should have its `existing_name` and `existing_description`
        set to the *enhanced* values that need validation.
        """
        return await self.validator.validate(data_element_for_validation)

    @cache_manager.async_cached(ttl=3600) # Cache results of this iterative process
    async def enhance_until_quality(self, 
                                  data_element: DataElement, # The original DataElement
                                  validation_feedback: str = "", # Initial feedback if any
                                  max_iterations: int = 1) -> Tuple[EnhancementResult, ValidationResult]:
        """
        Iteratively enhances a data element until it's deemed GOOD quality by the validator,
        or max_iterations are reached. Implements "enhance only if needed".

        Args:
            data_element: The original data element to enhance.
            validation_feedback: Initial validation feedback on the original element.
            max_iterations: Max enhancement attempts by this method. (Note: LLM might do internal "iterations" too)

        Returns:
            A tuple containing the final EnhancementResult and the ValidationResult of that final enhancement.
        """
        current_element_to_enhance = data_element # Start with the original
        feedback_for_llm = validation_feedback # Initial feedback for the first LLM call

        final_enhancement_attempt = None
        final_validation_of_attempt = None

        for i in range(max_iterations):
            iteration_num = i + 1
            logger.info(f"EnhancerAgent.enhance_until_quality: Iteration {iteration_num}/{max_iterations} for element ID: {data_element.id}, Current Name: '{current_element_to_enhance.existing_name}'")

            # Call the core enhancement logic (which itself might be one LLM call)
            # This `enhance` call gets the current version of the element and feedback.
            current_enhancement_result = await self.enhance(
                current_element_to_enhance, 
                feedback_for_llm
            )
            final_enhancement_attempt = current_enhancement_result # Store this as the latest attempt

            # If the enhancer decided not to change (due to "Enhance Only If Needed"),
            # its output name/desc will be same as current_element_to_enhance's input.
            # We still need to validate this version.
            element_version_after_enhancement_logic = DataElement(
                id=data_element.id,
                existing_name=current_enhancement_result.enhanced_name, # Key: use the name from LLM's "Enhanced Name" field
                existing_description=current_enhancement_result.enhanced_description, # Key: use the desc from LLM's "Enhanced Description"
                example=data_element.example,
                processes=data_element.processes,
                cdm=data_element.cdm
            )

            # Validate the output of the enhancement logic
            validation_of_this_enhancement = await self.validate_enhanced_element(element_version_after_enhancement_logic)
            final_validation_of_attempt = validation_of_this_enhancement

            if validation_of_this_enhancement.quality_status == DataQualityStatus.GOOD:
                logger.info(f"EnhancerAgent: Element {data_element.id} reached GOOD quality after iteration {iteration_num}.")
                # If no changes were made by LLM and it's GOOD, this means original was good.
                if (current_enhancement_result.enhanced_name == current_element_to_enhance.existing_name and
                    current_enhancement_result.enhanced_description == current_element_to_enhance.existing_description):
                    final_enhancement_attempt.feedback = "Original name and description met quality standards; no changes were made by the enhancement logic."
                    final_enhancement_attempt.confidence = max(final_enhancement_attempt.confidence, 0.95)
                break # Exit loop, quality is GOOD

            # Prepare for next iteration (if any)
            # The element to enhance next time is the *output* of the current enhancement attempt.
            current_element_to_enhance = element_version_after_enhancement_logic
            feedback_for_llm = "Feedback on previous enhancement attempt (Quality: {}):\n{}".format(
                validation_of_this_enhancement.quality_status.value,
                validation_of_this_enhancement.feedback
            )
            if validation_of_this_enhancement.suggested_improvements:
                feedback_for_llm += "\nFurther suggestions for this attempt:\n- " + "\n- ".join(validation_of_this_enhancement.suggested_improvements)
            
            logger.info(f"EnhancerAgent: Iteration {iteration_num} for {data_element.id} did not achieve GOOD. Quality of attempt: {validation_of_this_enhancement.quality_status.value}.")

        else: # Loop completed all iterations without break
            logger.warning(f"EnhancerAgent: Max iterations ({max_iterations}) reached for {data_element.id}. Final quality of last attempt: {final_validation_of_attempt.quality_status.value if final_validation_of_attempt else 'N/A'}")

        # Ensure we return valid objects even if loop didn't run or broke early
        if final_enhancement_attempt is None: # Should only happen if max_iterations = 0
            final_enhancement_attempt = EnhancementResult(
                enhanced_name=data_element.existing_name, 
                enhanced_description=data_element.existing_description,
                feedback="No enhancement iterations were performed.",
                confidence=0.1 # Low confidence as no work was done
            )
        if final_validation_of_attempt is None: # If loop didn't run, validate the original
             # This creates a DataElement with original name/desc in "existing" fields for validation
            final_validation_of_attempt = await self.validate_enhanced_element(data_element)
            
        return final_enhancement_attempt, final_validation_of_attempt

    @cache_manager.async_cached(ttl=3600) # Cache individual enhance calls
    async def enhance(self, data_element: DataElement, validation_feedback: str = "") -> EnhancementResult:
        """
        Performs a single enhancement attempt on the data element using the LLM.
        This is the core call to the LLM for enhancement.
        """
        try:
            acronym_info_str = self._prepare_acronym_info(data_element)
            processes_info_str = self._format_processes_info(data_element)

            llm_response_str = await self.enhancement_chain.ainvoke({
                "id": data_element.id,
                "name": data_element.existing_name or "", # Current name to be enhanced
                "description": data_element.existing_description or "", # Current desc to be enhanced
                "example": data_element.example or "Not provided.",
                "processes_info": processes_info_str,
                "validation_feedback": validation_feedback or "No specific prior validation feedback for this attempt. Evaluate current name/description from scratch.",
                "acronym_info": acronym_info_str,
            })
            
            enhancement_result = self._parse_enhancement_result(llm_response_str)
            
            logger.debug(f"EnhancerAgent.enhance (single LLM call) for {data_element.id}: Name='{enhancement_result.enhanced_name}', Desc='{enhancement_result.enhanced_description[:50]}...', Conf={enhancement_result.confidence:.2f}")
            
            # Check if parsing failed significantly, log and provide fallback
            if enhancement_result.enhanced_name == "parsing_error_name" or enhancement_result.enhanced_description.startswith("Parsing error:"):
                 logger.warning(f"EnhancerAgent: Parsing of LLM output failed for element {data_element.id}. Raw LLM: '{llm_response_str[:200]}...'")
                 # Fallback to original if enhancement is clearly broken
                 enhancement_result.enhanced_name = data_element.existing_name if enhancement_result.enhanced_name == "parsing_error_name" else enhancement_result.enhanced_name
                 if enhancement_result.enhanced_description.startswith("Parsing error:"):
                    enhancement_result.enhanced_description = data_element.existing_description
                 enhancement_result.feedback += " Warning: LLM output parsing may have failed. Result reflects original or partially parsed data."
                 enhancement_result.confidence = 0.1 # Low confidence if parsing failed
            return enhancement_result
        except Exception as e:
            logger.error(f"Error in EnhancerAgent.enhance for data element {data_element.id}: {e}", exc_info=True)
            # Return original data with error feedback in case of system error
            return EnhancementResult(
                enhanced_name=data_element.existing_name or "error_name", 
                enhanced_description=data_element.existing_description or "Error in enhancement.",
                feedback=f"System error during enhancement: {str(e)}",
                confidence=0.0
            )

    async def batch_enhance(self, data_elements: List[DataElement], validation_feedback: str = "") -> List[EnhancementResult]:
        """
        Enhances multiple data elements in parallel (each undergoing a single enhancement attempt).
        """
        tasks = [self.enhance(element, validation_feedback) for element in data_elements]
        return await asyncio.gather(*tasks)
