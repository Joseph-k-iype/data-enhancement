"""
OptimizedDataEnhancementWorkflow: Manages the overall process of enhancing a data element.
It now primarily uses the EnhancerAgent's iterative `enhance_until_quality` method.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, TypedDict, AsyncGenerator
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END, START

from app.core.models import (
    DataElement,
    EnhancedDataElement,
    ValidationResult, # For typing
    EnhancementResult, # For typing
    DataQualityStatus,
    Process # For typing
)
from app.utils.cache import cache_manager
from app.agents.validator_agent import ValidatorAgent
from app.agents.enhancer_agent import EnhancerAgent

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """
    State for the data enhancement workflow.
    """
    original_data_element: DataElement # The initial DataElement model passed to the workflow
    
    # The final result of the enhancement process
    final_enhanced_data_model: Optional[EnhancedDataElement] 
    
    # For tracking and control (though less critical now as EnhancerAgent handles iterations)
    workflow_iterations_done: int # How many times the main workflow step ran (typically 1)
    max_workflow_iterations: int  # Max times the main workflow step can run (typically 1)
    
    # Status and error tracking
    is_complete: bool
    error_message: Optional[str]


class OptimizedDataEnhancementWorkflow:
    """
    Manages the data enhancement process. It now primarily orchestrates a call 
    to the EnhancerAgent's `enhance_until_quality` method, which handles its own
    internal iterative refinement.
    """

    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        # ValidatorAgent is still needed for the dedicated /validate endpoint
        self.validator = ValidatorAgent(llm)
        # EnhancerAgent will perform the core iterative enhancement logic
        self.enhancer = EnhancerAgent(llm)
        self.graph = self._build_graph()

    async def _initialize_workflow_state_node(self, state: WorkflowState) -> WorkflowState:
        """
        Initializes the workflow state.
        """
        # `original_data_element` is passed in when invoking the graph.
        # `max_workflow_iterations` is also passed in.
        state["workflow_iterations_done"] = 0
        state["final_enhanced_data_model"] = None
        state["is_complete"] = False
        state["error_message"] = None
        logger.info(f"Workflow initialized for element: {state['original_data_element'].id}. Max workflow iterations: {state['max_workflow_iterations']}")
        return state

    async def _execute_enhancement_process_node(self, state: WorkflowState) -> WorkflowState:
        """
        This node executes the core enhancement logic by calling the EnhancerAgent's
        iterative `enhance_until_quality` method.
        """
        try:
            current_iteration = state["workflow_iterations_done"] + 1
            logger.info(f"Workflow: Starting main enhancement process (workflow iteration {current_iteration}/{state['max_workflow_iterations']}) for {state['original_data_element'].id}")

            # Initial validation of the original element to provide feedback to the enhancer
            # This is crucial if the enhancer's first step needs to know the current state.
            initial_validation_result = await self.validator.validate(state['original_data_element'])
            initial_feedback_for_enhancer = initial_validation_result.feedback
            if initial_validation_result.suggested_improvements:
                initial_feedback_for_enhancer += "\nInitial suggestions: " + "; ".join(initial_validation_result.suggested_improvements)

            if initial_validation_result.quality_status == DataQualityStatus.GOOD:
                logger.info(f"Workflow: Initial validation of {state['original_data_element'].id} is GOOD. Enhancer will confirm or make no changes.")
                # Enhancer's "Enhance Only If Needed" should handle this.
                # We can pass a stronger signal in feedback.
                initial_feedback_for_enhancer = "Initial assessment is GOOD. Confirm compliance and make no changes if already fully compliant with OPR, clarity, and layperson understandability."


            # Call the EnhancerAgent's iterative process.
            # The `enhance_until_quality` method handles its own internal loop.
            # The `max_iterations` for `enhance_until_quality` can be set here (e.g., 1 or 2 internal enhancer iterations).
            enhancement_result, validation_of_final_enhancement = await self.enhancer.enhance_until_quality(
                data_element=state['original_data_element'], # Always enhance based on the original
                validation_feedback=initial_feedback_for_enhancer,
                max_iterations=2 # Allow enhancer 1-2 internal attempts for refinement
            )

            # Construct the final EnhancedDataElement based on the results
            original_model = state['original_data_element']
            state["final_enhanced_data_model"] = EnhancedDataElement(
                id=original_model.id,
                existing_name=original_model.existing_name,
                existing_description=original_model.existing_description,
                example=original_model.example,
                processes=original_model.processes,
                cdm=original_model.cdm,
                enhanced_name=enhancement_result.enhanced_name,
                enhanced_description=enhancement_result.enhanced_description,
                quality_status=validation_of_final_enhancement.quality_status,
                enhancement_iterations=current_iteration, # This is workflow iterations. Enhancer tracks its own.
                validation_feedback=[validation_of_final_enhancement.feedback],
                enhancement_feedback=[enhancement_result.feedback],
                confidence_score=enhancement_result.confidence
            )
            
            state["workflow_iterations_done"] = current_iteration
            logger.info(f"Workflow: Main enhancement process for {original_model.id} complete. Final Quality: {validation_of_final_enhancement.quality_status.value}")
            return state

        except Exception as e:
            logger.error(f"Error in _execute_enhancement_process_node for {state['original_data_element'].id}: {e}", exc_info=True)
            state["error_message"] = f"Error during enhancement execution: {str(e)}"
            # is_complete will be set by _should_continue_or_finalize
            return state

    def _should_continue_or_finalize(self, state: WorkflowState) -> str:
        """
        Determines if the workflow should continue (e.g., for more top-level iterations, though current design is 1)
        or finalize.
        """
        if state.get("error_message"):
            logger.warning(f"Workflow: Error detected for {state['original_data_element'].id}, proceeding to finalize. Error: {state['error_message']}")
            return "finalize_workflow_node"

        # Since the enhancer agent now handles its own iterations to achieve GOOD quality,
        # this workflow typically runs the enhancement process once.
        if state["workflow_iterations_done"] >= state["max_workflow_iterations"]:
            logger.info(f"Workflow: Max workflow iterations ({state['max_workflow_iterations']}) reached for {state['original_data_element'].id}. Finalizing.")
            return "finalize_workflow_node"
        
        # This path would be for additional top-level workflow iterations if designed.
        # For now, it's unlikely to be taken if max_workflow_iterations is 1.
        logger.info(f"Workflow: Element {state['original_data_element'].id} - workflow iteration {state['workflow_iterations_done']} done. Proceeding to another workflow iteration (if configured).")
        return "execute_enhancement_node" 


    async def _finalize_workflow_node(self, state: WorkflowState) -> WorkflowState:
        """
        Finalizes the workflow, ensuring a result is always populated.
        """
        state["is_complete"] = True
        original_model = state["original_data_element"]
        
        if not state.get("final_enhanced_data_model"): # If enhancement node had an error before populating
            logger.error(f"Workflow finalizing with error for {original_model.id}: {state.get('error_message', 'Unknown error before final model creation')}")
            state["final_enhanced_data_model"] = EnhancedDataElement(
                id=original_model.id, existing_name=original_model.existing_name, existing_description=original_model.existing_description,
                example=original_model.example, processes=original_model.processes, cdm=original_model.cdm,
                enhanced_name=original_model.existing_name, # Fallback to original
                enhanced_description=original_model.existing_description, # Fallback to original
                quality_status=DataQualityStatus.POOR,
                enhancement_iterations=state["workflow_iterations_done"],
                validation_feedback=["Error during workflow execution."],
                enhancement_feedback=[state.get("error_message", "An unspecified error occurred.")],
                confidence_score=0.0
            )
        elif state.get("error_message"): # If error occurred but model was partially populated
            if state["final_enhanced_data_model"]:
                 state["final_enhanced_data_model"].quality_status = DataQualityStatus.POOR
                 state["final_enhanced_data_model"].enhancement_feedback.append(f"Workflow Error: {state['error_message']}")
                 state["final_enhanced_data_model"].confidence_score = min(state["final_enhanced_data_model"].confidence_score, 0.1)


        logger.info(f"Workflow finalized for element {original_model.id}. Iterations: {state['workflow_iterations_done']}. Error: {state.get('error_message')}. Final Quality: {state.get('final_enhanced_data_model').quality_status.value if state.get('final_enhanced_data_model') else 'N/A'}")
        return state

    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph for the enhancement workflow.
        The workflow is now simpler: Initialize -> Execute Enhancement (iterative via EnhancerAgent) -> Finalize.
        """
        workflow_graph = StateGraph(WorkflowState)
        workflow_graph.add_node("initialize_state_node", self._initialize_workflow_state_node)
        workflow_graph.add_node("execute_enhancement_node", self._execute_enhancement_process_node)
        workflow_graph.add_node("finalize_workflow_node", self._finalize_workflow_node)

        workflow_graph.set_entry_point("initialize_state_node")
        workflow_graph.add_edge("initialize_state_node", "execute_enhancement_node")
        
        workflow_graph.add_conditional_edges(
            "execute_enhancement_node",
            self._should_continue_or_finalize,
            {
                # This conditional logic determines if more top-level workflow iterations are needed.
                # Given current design (enhancer handles its own loop), this usually goes to finalize.
                "execute_enhancement_node": "execute_enhancement_node", # If more workflow iterations
                "finalize_workflow_node": "finalize_workflow_node"    # If workflow iterations complete or error
            }
        )
        workflow_graph.add_edge("finalize_workflow_node", END)
        return workflow_graph.compile()

    @cache_manager.async_cached(ttl=3600) # Cache the final result of the entire workflow run
    async def run(self, data_element: DataElement, max_iterations: int = 1) -> EnhancedDataElement:
        """
        Runs the entire data enhancement workflow.
        Args:
            data_element: The DataElement to enhance.
            max_iterations: Maximum number of top-level workflow iterations (typically 1, as
                            EnhancerAgent handles its own internal iterations).
        """
        logger.info(f"Workflow run: Element ID: {data_element.id}, Max top-level workflow iterations: {max_iterations}")
        
        # `data_element` is the Pydantic model. `max_iterations` refers to workflow iterations.
        initial_graph_input = WorkflowState(original_data_element=data_element, max_workflow_iterations=max_iterations) # type: ignore
        
        final_graph_state = await self.graph.ainvoke(initial_graph_input)

        final_result_model = final_graph_state.get("final_enhanced_data_model")
        
        if not final_result_model:
            logger.critical(f"Workflow for {data_element.id} ended without final_enhanced_data_model. This indicates an issue in the graph finalization logic.")
            # Construct a minimal error response
            return EnhancedDataElement(
                 **data_element.dict(), # Spread original data
                 enhanced_name=data_element.existing_name, 
                 enhanced_description=data_element.existing_description,
                 quality_status=DataQualityStatus.POOR,
                 enhancement_feedback=["Critical error: Workflow finished without a final result model."],
                 confidence_score=0.0
            )
        
        logger.info(f"Workflow run: Completed for {data_element.id}. Final Quality: {final_result_model.quality_status.value}, Confidence: {final_result_model.confidence_score:.2f}")
        return final_result_model


    async def stream_run(self, data_element: DataElement, max_iterations: int = 1) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Runs the workflow and streams updates.
        max_iterations here refers to top-level workflow iterations.
        """
        logger.info(f"Workflow stream_run: Element ID: {data_element.id}, Max top-level workflow iterations: {max_iterations}")
        
        initial_graph_input = WorkflowState(original_data_element=data_element, max_workflow_iterations=max_iterations) # type: ignore

        yield {
            "status": "starting", "message": "Workflow initialized.", 
            "workflow_iteration": 0, "progress": 0.0, "element_id": data_element.id
        }
        
        current_graph_state = initial_graph_input.copy()

        async for event_output in self.graph.astream(initial_graph_input, {"recursion_limit": (max_iterations * 2) + 5}): # Adjusted recursion limit
            node_name = list(event_output.keys())[0]
            node_state_after_execution = event_output[node_name]
            current_graph_state.update(node_state_after_execution) # Keep local state tracker updated

            workflow_iters_done = current_graph_state.get("workflow_iterations_done", 0)
            progress = min(1.0, (workflow_iters_done / max_iterations) if max_iterations > 0 else 1.0)
            
            if current_graph_state.get("error_message"):
                yield { "status": "error", "message": f"Error during node '{node_name}': {current_graph_state['error_message']}",
                        "workflow_iteration": workflow_iters_done, "progress": progress, "element_id": data_element.id }
                return

            if node_name == "execute_enhancement_node": # After the main enhancement processing
                final_model_so_far = current_graph_state.get("final_enhanced_data_model") # This is populated by the node
                if final_model_so_far:
                    yield {
                        "status": "in_progress",
                        "message": f"Enhancement process completed. Quality of result: {final_model_so_far.quality_status.value}.",
                        "workflow_iteration": workflow_iters_done, "progress": progress,
                        "current_result": { # Provide snapshot of the current best enhanced version
                            "enhanced_name": final_model_so_far.enhanced_name,
                            "enhanced_description": final_model_so_far.enhanced_description,
                            "quality_status": final_model_so_far.quality_status.value,
                            "confidence_score": final_model_so_far.confidence_score
                        },
                        "element_id": data_element.id
                    }
            
            if node_name == "finalize_workflow_node" or current_graph_state.get("is_complete"):
                final_model = current_graph_state.get("final_enhanced_data_model")
                if final_model:
                    yield { "status": "completed", "message": "Workflow finalized.",
                            "workflow_iteration": workflow_iters_done, "progress": 1.0,
                            "result": final_model.dict(), "element_id": data_element.id }
                else: # Should be populated by finalize_node
                     yield { "status": "error", "message": "Workflow finalized but no final_enhanced_data_model found.",
                             "workflow_iteration": workflow_iters_done, "progress": 1.0, "element_id": data_element.id }
                logger.info(f"Workflow stream_run: Completed for element ID: {data_element.id}")
                return
        
        # Fallback if graph stream ends without hitting a clear "finalize_node" or "is_complete"
        logger.warning(f"Workflow stream_run for {data_element.id} finished graph iteration unexpectedly. Last state: {current_graph_state.get('is_complete')}")
        final_model_fallback = current_graph_state.get("final_enhanced_data_model")
        if final_model_fallback:
             yield { "status": "completed", "message": "Workflow finished (stream loop ended).",
                     "workflow_iteration": current_graph_state.get("workflow_iterations_done", max_iterations), "progress": 1.0,
                     "result": final_model_fallback.dict(), "element_id": data_element.id }
        else: # Should have been populated by finalize or error handling
            yield { "status": "error", "message": "Workflow stream ended without final result model.",
                    "workflow_iteration": current_graph_state.get("workflow_iterations_done", max_iterations), "progress": 1.0,
                    "element_id": data_element.id }

# Alias for backward compatibility if any other module imports this specific name
DataEnhancementWorkflow = OptimizedDataEnhancementWorkflow
