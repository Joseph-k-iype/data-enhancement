"""
Improved workflow that combines validation and enhancement in fewer LLM calls.
Replace app/agents/workflow.py with this optimized version.
"""

import logging
import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated, AsyncGenerator
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.core.models import (
    DataElement, 
    EnhancedDataElement, 
    ValidationResult, 
    EnhancementResult, 
    DataQualityStatus,
    Process
)
from app.utils.cache import cache_manager

logger = logging.getLogger(__name__)

# Define workflow state
class WorkflowState(TypedDict):
    """State for the data enhancement workflow."""
    data_element: Dict[str, Any]
    enhanced_data: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    enhancement_result: Optional[Dict[str, Any]]
    iterations: int
    max_iterations: int
    is_complete: bool
    error: Optional[str]

class OptimizedDataEnhancementWorkflow:
    """Optimized LangGraph workflow for enhancing data elements with reduced LLM calls."""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()
        
        # Set up the single-call prompt for both validation and enhancement
        self.combined_prompt = PromptTemplate(
            input_variables=["id", "name", "description", "example", "processes_info"],
            template="""
            You are an expert in data governance and ISO/IEC 11179 metadata standards. Your task is to:
            1. VALIDATE the given data element name and description against ISO standards
            2. ENHANCE the data element to meet these standards
            
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
            
            Data Element to Evaluate and Enhance:
            - ID: {id}
            - Current Name: {name}
            - Current Description: {description}
            - Example (if provided): {example}
            {processes_info}
            
            First, provide a VALIDATION of the current data element with:
            1. Name valid: [yes/no]
            2. Name feedback: [details]
            3. Description valid: [yes/no]
            4. Description feedback: [details]
            5. Overall quality (GOOD, NEEDS_IMPROVEMENT, or POOR): [quality]
            6. Improvements needed: [list]
            
            Then, provide an ENHANCEMENT of the data element with:
            1. Enhanced Name: [improved name]
            2. Enhanced Description: [improved description]
            3. Enhancement Notes: [explanation]
            4. Confidence Score (0.0-1.0): [score]
            
            Your response MUST include both sections (VALIDATION and ENHANCEMENT).
            """
        )
        self.combined_chain = self.combined_prompt | self.llm | StrOutputParser()
    
    def _convert_processes_to_model(self, processes_data):
        """Safely convert process data to Process objects if needed."""
        if not processes_data:
            return None
            
        processes_list = []
        for proc in processes_data:
            # If already a Process object, use directly
            if isinstance(proc, Process):
                processes_list.append(proc)
            # If a dict, convert to Process object
            elif isinstance(proc, dict):
                processes_list.append(Process(**proc))
            else:
                logger.warning(f"Unknown process type: {type(proc)}")
        
        return processes_list
    
    def _format_processes_info(self, data_element: DataElement) -> str:
        """Format the processes information for the prompt."""
        if not data_element.processes:
            return "Related Processes: None"
        
        processes = data_element.processes
        # Ensure processes are Process objects
        if not all(isinstance(p, Process) for p in processes):
            # Convert dictionaries to Process objects
            processes = self._convert_processes_to_model(data_element.processes)
        
        processes_info = "Related Processes:\n"
        
        for i, process in enumerate(processes, 1):
            processes_info += f"Process {i} ID: {process.process_id}\n"
            processes_info += f"Process {i} Name: {process.process_name}\n"
            if process.process_description:
                processes_info += f"Process {i} Description: {process.process_description}\n"
            processes_info += "\n"
        
        return processes_info
    
    @cache_manager.async_cached(ttl=3600)  # Cache for 1 hour
    async def _single_call_enhancement(self, state: WorkflowState) -> WorkflowState:
        """Perform validation and enhancement in a single LLM call to reduce latency."""
        try:
            # Handle processes correctly
            data_element_dict = state["data_element"].copy()
            
            # Convert processes if present
            if data_element_dict.get("processes"):
                data_element_dict["processes"] = self._convert_processes_to_model(data_element_dict.get("processes", []))
            
            original_element = DataElement(**data_element_dict)
            
            # Format processes information
            processes_info = self._format_processes_info(original_element)
            
            # Perform the combined validation and enhancement in a single call
            start_time = asyncio.get_event_loop().time()
            combined_result = await self.combined_chain.ainvoke({
                "id": original_element.id,
                "name": original_element.existing_name,
                "description": original_element.existing_description,
                "example": original_element.example or "Not provided",
                "processes_info": processes_info
            })
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Combined validation and enhancement completed in {elapsed:.2f}s")
            
            # Parse the validation and enhancement results
            validation_result = self._parse_validation_section(combined_result)
            enhancement_result = self._parse_enhancement_section(combined_result)
            
            # Update state with results
            state["validation_result"] = validation_result.dict()
            state["enhancement_result"] = enhancement_result.dict()
            
            # Set enhanced data
            processes_dicts = None
            if original_element.processes:
                processes_dicts = [proc.dict() for proc in original_element.processes]
            
            # Calculate confidence score based on quality status
            confidence_score = enhancement_result.confidence
            if validation_result.quality_status == DataQualityStatus.GOOD:
                confidence_score = max(confidence_score, 0.9)  # Boost confidence for good quality
            
            enhanced_data = {
                **state["data_element"],
                "enhanced_name": enhancement_result.enhanced_name,
                "enhanced_description": enhancement_result.enhanced_description,
                "processes": processes_dicts,
                "quality_status": validation_result.quality_status,
                "enhancement_iterations": 1,
                "enhancement_feedback": [enhancement_result.feedback],
                "validation_feedback": [validation_result.feedback],
                "confidence_score": confidence_score
            }
            state["enhanced_data"] = enhanced_data
            
            # Increment iteration counter
            state["iterations"] = state["iterations"] + 1
            
            # Check if we've achieved good quality
            if validation_result.quality_status == DataQualityStatus.GOOD:
                logger.info(f"Achieved GOOD quality on first attempt for element: {original_element.id}")
                state["is_complete"] = True
            
            return state
        except Exception as e:
            logger.error(f"Error in combined validation/enhancement: {e}")
            state["error"] = f"Error in validation/enhancement: {str(e)}"
            state["is_complete"] = True
            return state
    
    def _parse_validation_section(self, result: str) -> ValidationResult:
        """Parse the validation section of the combined result."""
        # Split by lines
        lines = result.split("\n")
        
        # Find the validation section
        validation_start = None
        validation_end = None
        
        for i, line in enumerate(lines):
            if "VALIDATION" in line.upper():
                validation_start = i
            elif validation_start and "ENHANCEMENT" in line.upper():
                validation_end = i
                break
        
        if validation_start is None:
            return ValidationResult(
                is_valid=False,
                quality_status=DataQualityStatus.POOR,
                feedback="Failed to parse validation results",
                suggested_improvements=[]
            )
        
        if validation_end is None:
            validation_end = len(lines)
        
        validation_text = "\n".join(lines[validation_start:validation_end])
        
        # Extract validation details
        name_valid = False
        for line in validation_text.split("\n"):
            if "name valid" in line.lower() and ("yes" in line.lower() or "valid" in line.lower()):
                name_valid = True
                break
                
        desc_valid = False
        for line in validation_text.split("\n"):
            if "description valid" in line.lower() and ("yes" in line.lower() or "valid" in line.lower()):
                desc_valid = True
                break
        
        # Extract quality status
        quality_status = DataQualityStatus.NEEDS_IMPROVEMENT
        if "GOOD" in validation_text.upper():
            quality_status = DataQualityStatus.GOOD
        elif "POOR" in validation_text.upper():
            quality_status = DataQualityStatus.POOR
        
        # Extract feedback
        feedback = "Name feedback: "
        name_feedback_match = re.search(r"Name feedback:(.+?)(?:\n\d+\.|\n\n|$)", validation_text, re.DOTALL)
        if name_feedback_match:
            feedback += name_feedback_match.group(1).strip()
        
        feedback += "\n\nDescription feedback: "
        desc_feedback_match = re.search(r"Description feedback:(.+?)(?:\n\d+\.|\n\n|$)", validation_text, re.DOTALL)
        if desc_feedback_match:
            feedback += desc_feedback_match.group(1).strip()
        
        # Extract improvements
        improvements = []
        improvements_section = re.search(r"Improvements needed:(.+?)(?:\n\n|ENHANCEMENT|$)", validation_text, re.DOTALL)
        if improvements_section:
            improvements_text = improvements_section.group(1).strip()
            for line in improvements_text.split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("*") or re.match(r"^\d+\.", line)):
                    # Remove the list marker
                    clean_line = re.sub(r"^[-*]\s*|\d+\.\s*", "", line).strip()
                    improvements.append(clean_line)
                elif line and improvements:
                    improvements[-1] += " " + line
        
        return ValidationResult(
            is_valid=name_valid and desc_valid,
            quality_status=quality_status,
            feedback=feedback,
            suggested_improvements=improvements
        )
    
    def _parse_enhancement_section(self, result: str) -> EnhancementResult:
        """Parse the enhancement section of the combined result."""
        # Split by lines
        lines = result.split("\n")
        
        # Find the enhancement section
        enhancement_start = None
        
        for i, line in enumerate(lines):
            if "ENHANCEMENT" in line.upper():
                enhancement_start = i
                break
        
        if enhancement_start is None:
            return EnhancementResult(
                enhanced_name="",
                enhanced_description="",
                feedback="Failed to parse enhancement results",
                confidence=0.0
            )
        
        enhancement_text = "\n".join(lines[enhancement_start:])
        
        # Extract enhanced name
        enhanced_name = ""
        name_match = re.search(r"Enhanced Name:(.+?)(?:\n\d+\.|\n\n|$)", enhancement_text, re.DOTALL)
        if name_match:
            enhanced_name = name_match.group(1).strip()
            # Remove any enclosing brackets or quotes
            enhanced_name = re.sub(r"^\[|\]$|^\"|\"\$", "", enhanced_name).strip()
        
        # Extract enhanced description
        enhanced_description = ""
        desc_match = re.search(r"Enhanced Description:(.+?)(?:\n\d+\.|\n\n|Enhancement Notes:|$)", enhancement_text, re.DOTALL)
        if desc_match:
            enhanced_description = desc_match.group(1).strip()
            # Remove any enclosing brackets or quotes
            enhanced_description = re.sub(r"^\[|\]$|^\"|\"\$", "", enhanced_description).strip()
        
        # Extract feedback
        feedback = ""
        feedback_match = re.search(r"Enhancement Notes:(.+?)(?:\n\d+\.|\n\n|Confidence Score|$)", enhancement_text, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            # Remove any enclosing brackets or quotes
            feedback = re.sub(r"^\[|\]$|^\"|\"\$", "", feedback).strip()
        
        # Extract confidence
        confidence = 0.7  # Default
        confidence_match = re.search(r"Confidence Score.+?(\d+\.\d+)", enhancement_text)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Ensure within bounds
            except ValueError:
                pass
        
        return EnhancementResult(
            enhanced_name=enhanced_name,
            enhanced_description=enhanced_description,
            feedback=feedback,
            confidence=confidence
        )
    
    def _should_continue(self, state: WorkflowState) -> str:
        """Determine if the process should continue or complete."""
        # Check if we've hit the maximum number of iterations
        if state["iterations"] >= state["max_iterations"]:
            logger.info(f"Reached maximum iterations ({state['max_iterations']})")
            return "complete"
        
        # Check if we've achieved good quality
        quality_status = state.get("validation_result", {}).get("quality_status")
        if quality_status == DataQualityStatus.GOOD:
            logger.info("Quality status is GOOD, completing workflow")
            return "complete"
        
        # If we have enhanced data, check its quality status
        if state.get("enhanced_data") and state["enhanced_data"].get("quality_status") == DataQualityStatus.GOOD:
            logger.info("Enhanced data quality status is GOOD, completing workflow")
            return "complete"
        
        # Continue the process only if quality is not good and we haven't reached max iterations
        logger.info(f"Quality status is {quality_status}, continuing enhancement")
        return "continue"
    
    async def _complete_workflow(self, state: WorkflowState) -> WorkflowState:
        """Mark the workflow as complete."""
        state["is_complete"] = True
        logger.info("Workflow completed")
        
        # Add a final confidence evaluation if needed
        if state.get("enhanced_data") and state["enhanced_data"]["quality_status"] == DataQualityStatus.GOOD:
            logger.info("Enhancement achieved GOOD quality - setting high confidence")
            state["enhanced_data"]["confidence_score"] = 1.0
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build the optimized LangGraph workflow."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("single_call_enhancement", self._single_call_enhancement)
        workflow.add_node("complete", self._complete_workflow)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "single_call_enhancement",
            self._should_continue,
            {
                "continue": "single_call_enhancement",
                "complete": "complete"
            }
        )
        
        # Add edge to end
        workflow.add_edge("complete", END)
        
        # Set entrypoint
        workflow.set_entry_point("single_call_enhancement")
        
        return workflow.compile()
    
    @cache_manager.async_cached(ttl=3600)  # Cache for 1 hour
    async def run(self, data_element: DataElement, max_iterations: int = 2) -> EnhancedDataElement:
        """
        Run the data enhancement workflow on a data element.
        
        Args:
            data_element: The data element to enhance
            max_iterations: Maximum number of enhancement iterations to perform (default reduced to 2)
        
        Returns:
            Enhanced data element
        """
        logger.info(f"Starting enhancement workflow for element: {data_element.id}")
        
        # Convert processes to dict for serializing in the state
        element_dict = data_element.dict()
        if data_element.processes:
            element_dict["processes"] = [proc.dict() for proc in data_element.processes]
        
        initial_state: WorkflowState = {
            "data_element": element_dict,
            "enhanced_data": None,
            "validation_result": None,
            "enhancement_result": None,
            "iterations": 0,
            "max_iterations": max_iterations,
            "is_complete": False,
            "error": None
        }
        
        # Run the workflow
        result = await self.graph.ainvoke(initial_state)
        
        if result.get("error"):
            logger.error(f"Workflow error: {result['error']}")
            raise ValueError(result["error"])
        
        if not result.get("enhanced_data"):
            logger.info("No enhancement was performed, using original data")
            # If no enhancement was performed, create an enhanced data element from the original
            return EnhancedDataElement(
                **data_element.dict(),
                enhanced_name=data_element.existing_name,
                enhanced_description=data_element.existing_description,
                quality_status=result["validation_result"]["quality_status"] if result.get("validation_result") else DataQualityStatus.GOOD,
                enhancement_iterations=0,
                validation_feedback=[result["validation_result"]["feedback"]] if result.get("validation_result") else [],
                confidence_score=0.5  # Default confidence
            )
        
        # Handle the processes correctly, converting dict to Process objects if needed
        if result["enhanced_data"].get("processes"):
            result["enhanced_data"]["processes"] = self._convert_processes_to_model(result["enhanced_data"]["processes"])
        
        # Convert the enhanced data dict back to an EnhancedDataElement
        logger.info(f"Workflow completed with {result['enhanced_data'].get('enhancement_iterations', 0)} iterations")
        return EnhancedDataElement(**result["enhanced_data"])
    
    async def stream_run(self, data_element: DataElement, max_iterations: int = 2) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the workflow and stream results after each iteration.
        Max iterations reduced to 2 for better performance.
        """
        # Convert processes to dict for serializing in the state
        element_dict = data_element.dict()
        if data_element.processes:
            element_dict["processes"] = [proc.dict() for proc in data_element.processes]
        
        initial_state: WorkflowState = {
            "data_element": element_dict,
            "enhanced_data": None,
            "validation_result": None,
            "enhancement_result": None,
            "iterations": 0,
            "max_iterations": max_iterations,
            "is_complete": False,
            "error": None
        }
        
        # Manually control the workflow for streaming
        state = initial_state
        
        # Initial progress update
        yield {
            "status": "in_progress",
            "message": "Starting enhancement workflow",
            "iteration": 0,
            "progress": 0.0
        }
        
        try:
            # Run first iteration
            state = await self._single_call_enhancement(state)
            
            if state.get("error"):
                yield {
                    "status": "error",
                    "message": state["error"],
                    "iteration": state["iterations"],
                    "progress": 0.0
                }
                return
            
            # Send first iteration results
            progress = min(1.0, state["iterations"] / max_iterations)
            yield {
                "status": "in_progress",
                "message": f"Completed iteration {state['iterations']}",
                "iteration": state["iterations"],
                "progress": progress,
                "current_result": {
                    "enhanced_name": state["enhanced_data"]["enhanced_name"],
                    "enhanced_description": state["enhanced_data"]["enhanced_description"],
                    "quality_status": state["enhanced_data"]["quality_status"],
                    "confidence_score": state["enhanced_data"]["confidence_score"]
                }
            }
            
            # Continue iterations until complete or hit limit
            while not state["is_complete"] and state["iterations"] < max_iterations:
                if self._should_continue(state) == "complete":
                    state = await self._complete_workflow(state)
                    break
                
                # Run next iteration
                state = await self._single_call_enhancement(state)
                
                if state.get("error"):
                    yield {
                        "status": "error",
                        "message": state["error"],
                        "iteration": state["iterations"],
                        "progress": progress
                    }
                    return
                
                # Send iteration results
                progress = min(1.0, state["iterations"] / max_iterations)
                yield {
                    "status": "in_progress",
                    "message": f"Completed iteration {state['iterations']}",
                    "iteration": state["iterations"],
                    "progress": progress,
                    "current_result": {
                        "enhanced_name": state["enhanced_data"]["enhanced_name"],
                        "enhanced_description": state["enhanced_data"]["enhanced_description"],
                        "quality_status": state["enhanced_data"]["quality_status"],
                        "confidence_score": state["enhanced_data"]["confidence_score"]
                    }
                }
            
            # Complete the workflow if not already
            if not state["is_complete"]:
                state = await self._complete_workflow(state)
            
            # Handle the processes correctly
            if state["enhanced_data"].get("processes"):
                state["enhanced_data"]["processes"] = self._convert_processes_to_model(state["enhanced_data"]["processes"])
            
            # Convert to EnhancedDataElement
            enhanced_element = EnhancedDataElement(**state["enhanced_data"])
            
            # Send final result
            yield {
                "status": "completed",
                "message": "Enhancement workflow completed",
                "iteration": state["iterations"],
                "progress": 1.0,
                "result": enhanced_element.dict()
            }
            
        except Exception as e:
            logger.error(f"Error in streaming workflow: {e}")
            yield {
                "status": "error",
                "message": f"Error: {str(e)}",
                "iteration": state.get("iterations", 0),
                "progress": min(1.0, state.get("iterations", 0) / max_iterations)
            }

# For backwards compatibility
DataEnhancementWorkflow = OptimizedDataEnhancementWorkflow