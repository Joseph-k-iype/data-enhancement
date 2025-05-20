import re
from typing import Dict, List, Any, Tuple

class ISO11179Validator:
    """Utility class for validating data elements against ISO/IEC 11179 standards."""
    
    @staticmethod
    def validate_name(name: str) -> Tuple[bool, str]:
        """
        Validate a data element name against ISO/IEC 11179 naming standards.
        Adapted for business-friendly, lowercase, space-separated names.
        
        ISO/IEC 11179 naming guidelines include:
        - Use of clear, unambiguous terms
        - Avoidance of abbreviations (unless widely understood)
        - Descriptive but concise
        - Business-friendly with proper spacing
        
        Returns:
            Tuple[bool, str]: (is_valid, feedback)
        """
        if not name or len(name.strip()) == 0:
            return False, "Name cannot be empty"
        
        # Check for minimum length
        if len(name) < 3:
            return False, "Name is too short, should be at least 3 characters"
        
        # Check for maximum length (reasonable for business context)
        if len(name) > 100:
            return False, "Name is too long, should be at most 100 characters"
        
        # Check if name is all lowercase with spaces
        if not re.match(r'^[a-z][a-z0-9 ]+$', name):
            return False, "Name should be in lowercase with spaces between words"
        
        # Check for proper word separation with spaces
        if "  " in name:
            return False, "Name should not contain multiple consecutive spaces"
        
        # Check for common poor naming practices
        poor_terms = ['data', 'info', 'value', 'val', 'var', 'temp', 'tmp']
        for term in poor_terms:
            if name.lower() == term or name.lower().startswith(term + " "):
                return False, f"Name uses generic term '{term}' which is not specific enough"
        
        # Check for name components (at least 2 words for descriptiveness)
        if len(name.split()) < 2:
            return False, "Name should contain at least two words to be descriptive enough"
        
        return True, "Name is valid according to business-friendly standards"
    
    @staticmethod
    def validate_description(description: str) -> Tuple[bool, str]:
        """
        Validate a data element description against ISO/IEC 11179 description standards.
        
        ISO/IEC 11179 description guidelines include:
        - Clear, unambiguous definition
        - Completeness (covers the concept fully)
        - Precision (specific enough to distinguish from other concepts)
        - Objectivity (factual, not opinion-based)
        
        Returns:
            Tuple[bool, str]: (is_valid, feedback)
        """
        if not description or len(description.strip()) == 0:
            return False, "Description cannot be empty"
        
        # Check for minimum length
        if len(description) < 10:
            return False, "Description is too short, should be at least 10 characters"
        
        # Check for maximum length (reasonable for business context)
        if len(description) > 5000:
            return False, "Description is too long, should be at most 500 characters"
        
        # Check for sentence structure
        sentences = re.split(r'[.!?]+', description)
        valid_sentences = [s for s in sentences if len(s.strip()) > 0]
        
        if len(valid_sentences) < 1:
            return False, "Description should contain at least one complete sentence"
        
        # Check for starting with a capital letter
        if not description[0].isupper():
            return False, "Description should start with a capital letter"
        
        # Check for ending with proper punctuation
        if not description[-1] in ['.', '!', '?']:
            return False, "Description should end with proper punctuation"
        
        # Check for vague terms
        vague_terms = ['etc', 'and so on', 'and more', 'things', 'stuff']
        for term in vague_terms:
            if term in description.lower():
                return False, f"Description contains vague term '{term}', be more specific"
        
        # Check for redundant phrases
        if re.search(r'\bis\s+is\b|\bthe\s+the\b|\band\s+and\b', description.lower()):
            return False, "Description contains redundant phrases"
        
        return True, "Description is valid according to ISO/IEC 11179 standards"
    
    @staticmethod
    def evaluate_data_element(name: str, description: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate a data element against ISO/IEC 11179 standards.
        Returns a tuple with a boolean indicating if the element is valid,
        and a dictionary with detailed evaluation results.
        """
        name_valid, name_feedback = ISO11179Validator.validate_name(name)
        desc_valid, desc_feedback = ISO11179Validator.validate_description(description)
        
        is_valid = name_valid and desc_valid
        
        result = {
            "is_valid": is_valid,
            "name_validation": {
                "is_valid": name_valid,
                "feedback": name_feedback
            },
            "description_validation": {
                "is_valid": desc_valid,
                "feedback": desc_feedback
            }
        }
        
        return is_valid, result