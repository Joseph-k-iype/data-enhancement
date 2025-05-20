import logging
from typing import Dict, Any, List, Optional, Tuple, TypedDict, Annotated
from langchain_openai import AzureChatOpenAI
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
from app.agents.validator_agent import ValidatorAgent
from app.agents.enhancer_agent import EnhancerAgent
from app.agents.confidence_evaluator import ConfidenceEvaluator

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

class DataEnhancementWorkflow:
    """LangGraph workflow for enhancing data elements with quality feedback loop."""
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.validator = ValidatorAgent(llm)
        self.enhancer = EnhancerAgent(llm)
        self.confidence_evaluator = ConfidenceEvaluator(llm)
        self.graph = self._build_graph()
    
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
    
    async def _validate_data(self, state: WorkflowState) -> WorkflowState:
        """Validate the data element."""
        try:
            # If there's enhanced data, validate that
            if state.get("enhanced_data"):
                # Handle processes correctly
                processes = None
                if state["data_element"].get("processes"):
                    processes = self._convert_processes_to_model(state["data_element"].get("processes", []))
                
                data_to_validate = DataElement(
                    id=state["data_element"]["id"],
                    existing_name=state["enhanced_data"]["enhanced_name"],
                    existing_description=state["enhanced_data"]["enhanced_description"],
                    example=state["data_element"].get("example"),
                    processes=processes,
                    cdm=state["data_element"].get("cdm")
                )
            else:
                # Otherwise validate the original data
                # Handle processes correctly
                data_element_dict = state["data_element"].copy()
                
                # Convert processes if present
                if data_element_dict.get("processes"):
                    data_element_dict["processes"] = self._convert_processes_to_model(data_element_dict.get("processes", []))
                
                data_to_validate = DataElement(**data_element_dict)
            
            # Validate the data
            validation_result = await self.validator.validate(data_to_validate)
            
            # Update state with validation result
            state["validation_result"] = validation_result.dict()
            
            # If this is a subsequent iteration, update the enhanced data with validation feedback
            if state.get("enhanced_data") and state["iterations"] > 0:
                if "validation_feedback" not in state["enhanced_data"]:
                    state["enhanced_data"]["validation_feedback"] = []
                state["enhanced_data"]["validation_feedback"].append(validation_result.feedback)
                state["enhanced_data"]["quality_status"] = validation_result.quality_status
            
            return state
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            state["error"] = f"Validation error: {str(e)}"
            state["is_complete"] = True
            return state
    
    async def _enhance_data(self, state: WorkflowState) -> WorkflowState:
        """Enhance the data element using the quality feedback loop."""
        try:
            # Get validation feedback
            validation_feedback = state["validation_result"]["feedback"]
            if state["validation_result"].get("suggested_improvements"):
                validation_feedback += "\n\nSuggested improvements:\n" + "\n".join(
                    state["validation_result"]["suggested_improvements"]
                )
            
            # Handle processes correctly
            data_element_dict = state["data_element"].copy()
            
            # Convert processes if present
            if data_element_dict.get("processes"):
                data_element_dict["processes"] = self._convert_processes_to_model(data_element_dict.get("processes", []))
            
            # Create the DataElement object
            original_element = DataElement(**data_element_dict)
            
            # Determine max iterations based on remaining allowed iterations
            remaining_iterations = state["max_iterations"] - state["iterations"]
            quality_iterations = min(3, remaining_iterations)  # Use at most 3 iterations for quality improvement
            
            if quality_iterations > 1:
                # Use the enhance_until_quality method for better results
                enhancement_result, validation_result = await self.enhancer.enhance_until_quality(
                    data_element=original_element,
                    validation_feedback=validation_feedback,
                    additional_context="",
                    max_iterations=quality_iterations
                )
                # Add the used iterations to the state's iteration count (subtract 1 as we'll increment later)
                state["iterations"] += quality_iterations - 1
            else:
                # Just do a single enhancement if we're on the last iteration
                enhancement_result = await self.enhancer.enhance(
                    original_element, 
                    validation_feedback
                )
                # Validate the result to get quality status
                test_element = DataElement(
                    id=original_element.id,
                    existing_name=enhancement_result.enhanced_name,
                    existing_description=enhancement_result.enhanced_description,
                    example=original_element.example,
                    processes=original_element.processes,
                    cdm=original_element.cdm
                )
                validation_result = await self.validator.validate(test_element)
            
            state["enhancement_result"] = enhancement_result.dict()
            
            # Create or update enhanced data
            if not state.get("enhanced_data"):
                confidence_score = enhancement_result.confidence
                
                if state["iterations"] == 0:
                    # Confidence based on quality status for first iteration
                    if validation_result.quality_status == DataQualityStatus.GOOD:
                        confidence_score = 1.0
                    elif validation_result.quality_status == DataQualityStatus.NEEDS_IMPROVEMENT:
                        confidence_score = 0.8
                    else:
                        confidence_score = 0.5
                
                # Convert processes to dicts for serialization
                processes_dicts = None
                if original_element.processes:
                    processes_dicts = [proc.dict() for proc in original_element.processes]
                
                enhanced_data = {
                    **state["data_element"],
                    "enhanced_name": enhancement_result.enhanced_name,
                    "enhanced_description": enhancement_result.enhanced_description,
                    "processes": processes_dicts,  # Use serialized processes
                    "quality_status": validation_result.quality_status,
                    "enhancement_iterations": 1,
                    "enhancement_feedback": [enhancement_result.feedback],
                    "validation_feedback": [validation_result.feedback],
                    "confidence_score": confidence_score
                }
                state["enhanced_data"] = enhanced_data
            else:
                # Update the confidence score based on new quality status
                if validation_result.quality_status == DataQualityStatus.GOOD:
                    new_confidence = 1.0
                elif validation_result.quality_status == DataQualityStatus.NEEDS_IMPROVEMENT:
                    new_confidence = 0.8
                else:
                    new_confidence = 0.5
                
                state["enhanced_data"]["enhanced_name"] = enhancement_result.enhanced_name
                state["enhanced_data"]["enhanced_description"] = enhancement_result.enhanced_description
                state["enhanced_data"]["enhancement_iterations"] = state["enhanced_data"].get("enhancement_iterations", 0) + 1
                state["enhanced_data"]["confidence_score"] = new_confidence
                state["enhanced_data"]["quality_status"] = validation_result.quality_status
                
                if "enhancement_feedback" not in state["enhanced_data"]:
                    state["enhanced_data"]["enhancement_feedback"] = []
                state["enhanced_data"]["enhancement_feedback"].append(enhancement_result.feedback)
                
                if "validation_feedback" not in state["enhanced_data"]:
                    state["enhanced_data"]["validation_feedback"] = []
                state["enhanced_data"]["validation_feedback"].append(validation_result.feedback)
            
            # Increment iteration counter
            state["iterations"] = state["iterations"] + 1
            
            return state
        except Exception as e:
            logger.error(f"Enhancement error: {e}")
            state["error"] = f"Enhancement error: {str(e)}"
            state["is_complete"] = True
            return state
    
    def _should_enhance(self, state: WorkflowState) -> str:
        """Determine if the data needs further enhancement."""
        # Check if we've hit the maximum number of iterations
        if state["iterations"] >= state["max_iterations"]:
            logger.info(f"Reached maximum iterations ({state['max_iterations']})")
            return "complete"
        
        # Check validation result
        if state["validation_result"]["quality_status"] == DataQualityStatus.GOOD:
            logger.info("Quality status is GOOD, completing workflow")
            return "complete"
        
        # If we have enhanced data, check its quality status
        if state.get("enhanced_data") and state["enhanced_data"].get("quality_status") == DataQualityStatus.GOOD:
            logger.info("Enhanced data quality status is GOOD, completing workflow")
            return "complete"
        
        # Otherwise, continue enhancing
        logger.info(f"Quality status is {state['validation_result']['quality_status']}, continuing enhancement")
        return "enhance"
    
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
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("validate", self._validate_data)
        workflow.add_node("enhance", self._enhance_data)
        workflow.add_node("complete", self._complete_workflow)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate",
            self._should_enhance,
            {
                "enhance": "enhance",
                "complete": "complete"
            }
        )
        
        # Add remaining edges
        workflow.add_edge("enhance", "validate")
        workflow.add_edge("complete", END)  # End of workflow
        
        # Set entrypoint
        workflow.set_entry_point("validate")
        
        return workflow.compile()
    
    async def run(self, data_element: DataElement, max_iterations: int = 5) -> EnhancedDataElement:
        """
        Run the data enhancement workflow on a data element.
        
        Args:
            data_element: The data element to enhance
            max_iterations: Maximum number of enhancement iterations to perform
        
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