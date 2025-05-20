from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class DataQualityStatus(str, Enum):
    """Quality status of a data element."""
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

class EnhancementStatus(str, Enum):
    """Status of an enhancement request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Process(BaseModel):
    """Model representing a business process."""
    process_id: str = Field(..., description="Unique identifier for the process")
    process_name: str = Field(..., description="Name of the process")
    process_description: Optional[str] = Field(None, description="Description of the process")

class DataElement(BaseModel):
    """Model representing a data element with name and description."""
    id: str = Field(..., description="Unique identifier for the data element")
    existing_name: str = Field(..., description="Current name of the data element")
    existing_description: str = Field(..., description="Current description of the data element")
    example: Optional[str] = Field(None, description="Example of the data element")
    processes: Optional[List[Process]] = Field(None, description="Related processes")
    cdm: Optional[str] = Field(None, description="Common Data Model category for the data element")
    
    @field_validator('id')
    def id_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('ID must not be empty')
        return v.strip()
    
    @field_validator('existing_name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Name must not be empty')
        return v.strip()
    
    @field_validator('existing_description')
    def description_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Description must not be empty')
        return v.strip()

class EnhancedDataElement(DataElement):
    """Model representing an enhanced data element."""
    enhanced_name: str = Field(..., description="Enhanced name of the data element")
    enhanced_description: str = Field(..., description="Enhanced description of the data element")
    quality_status: DataQualityStatus = Field(DataQualityStatus.NEEDS_IMPROVEMENT, 
                                             description="Quality status of the data element")
    enhancement_iterations: int = Field(0, description="Number of enhancement iterations performed")
    validation_feedback: List[str] = Field(default_factory=list, 
                                          description="Feedback from validation iterations")
    enhancement_feedback: List[str] = Field(default_factory=list, 
                                           description="Feedback from enhancement iterations")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, 
                                    description="Confidence score for the enhancement")

class ValidationResult(BaseModel):
    """Model representing the result of data validation."""
    is_valid: bool = Field(..., description="Whether the data element is valid")
    quality_status: DataQualityStatus = Field(..., description="Quality status of the data element")
    feedback: str = Field(..., description="Feedback on the quality of the data element")
    suggested_improvements: Optional[List[str]] = Field(default_factory=list, 
                                                       description="Suggested improvements")

class EnhancementResult(BaseModel):
    """Model representing the result of data enhancement."""
    enhanced_name: str = Field(..., description="Enhanced name of the data element")
    enhanced_description: str = Field(..., description="Enhanced description of the data element")
    feedback: str = Field(..., description="Feedback on the enhancement process")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    @field_validator('confidence')
    def confidence_must_be_positive(cls, v):
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

class EnhancementRequest(BaseModel):
    """Model representing a request for data enhancement."""
    data_element: DataElement = Field(..., description="Data element to enhance")
    max_iterations: Optional[int] = Field(5, description="Maximum number of enhancement iterations")
    
class EnhancementResponse(BaseModel):
    """Model representing a response to an enhancement request."""
    request_id: str = Field(..., description="ID of the enhancement request")
    status: EnhancementStatus = Field(..., description="Status of the enhancement request")
    enhanced_data: Optional[EnhancedDataElement] = Field(None, 
                                                        description="Enhanced data element")
    error_message: Optional[str] = Field(None, description="Error message if enhancement failed")

# Note: All PBTTagging-related models have been removed