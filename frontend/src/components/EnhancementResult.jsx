// src/components/EnhancementResult.jsx
import React from 'react';

const EnhancementResult = ({ result }) => {
  if (!result) return null;
  
  const { 
    existing_name, 
    existing_description, 
    enhanced_name, 
    enhanced_description, 
    quality_status,
    confidence_score,
    enhancement_feedback
  } = result;

  // Calculate confidence percentage
  const confidencePercentage = Math.round(confidence_score * 100);
  
  // Determine badge class based on quality status
  const qualityClass = quality_status?.toLowerCase() || 'needs_improvement';

  return (
    <div className="enhancement-result">
      <div className={`quality-badge quality-${qualityClass}`}>
        {quality_status?.replace(/_/g, ' ') || 'Needs Improvement'}
      </div>
      
      <div className="comparison">
        <div className="comparison-card">
          <h4>Original Name</h4>
          <p>{existing_name}</p>
        </div>
        <div className="comparison-card">
          <h4>Enhanced Name</h4>
          <p>{enhanced_name}</p>
        </div>
        <div className="comparison-card">
          <h4>Original Description</h4>
          <p>{existing_description}</p>
        </div>
        <div className="comparison-card">
          <h4>Enhanced Description</h4>
          <p>{enhanced_description}</p>
        </div>
      </div>
      
      <div className="form-group">
        <label>Enhancement Feedback</label>
        <div className="feedback-box">
          {enhancement_feedback && enhancement_feedback.length > 0 ? (
            enhancement_feedback.map((feedback, index) => (
              <p key={index}>{feedback}</p>
            ))
          ) : (
            <p>The name and description have been enhanced according to ISO/IEC 11179 standards.</p>
          )}
        </div>
      </div>
      
      <div className="form-group">
        <label>Confidence Score</label>
        <div className="confidence-meter">
          <div className="confidence-bar">
            <div 
              className="confidence-fill" 
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
          <div className="confidence-value">{confidencePercentage}%</div>
        </div>
      </div>
      
      <div className="result-actions">
        <button className="btn btn-outline">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 10H16C17.7 10 19 11.3 19 13V14" stroke="currentColor" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M5 6H18C19.7 6 21 7.3 21 9V10" stroke="currentColor" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M11 22H4C3.4 22 3 21.6 3 21V15H12V21C12 21.6 11.6 22 11 22Z" stroke="currentColor" strokeWidth="1.5" strokeMiterlimit="10" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Copy to Clipboard
        </button>
        <button className="btn btn-primary">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 10.5L11 12.5L15.5 8" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2C6.5 2 2 6.5 2 12C2 17.5 6.5 22 12 22Z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Save Enhanced Element
        </button>
      </div>
    </div>
  );
};

export default EnhancementResult;