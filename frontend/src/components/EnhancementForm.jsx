// src/components/EnhancementForm.jsx
import React, { useState } from 'react';

const EnhancementForm = ({ onSubmit, loading, submitLabel = "Enhance Element" }) => {
  const [formData, setFormData] = useState({
    id: `DE-${Math.floor(Math.random() * 10000)}`,
    existing_name: '',
    existing_description: '',
    example: '',
    processes: [],
    cdm: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form onSubmit={handleSubmit} className="enhancement-form">
      <div className="form-group">
        <label htmlFor="id">Element ID</label>
        <input
          type="text"
          id="id"
          name="id"
          className="form-control"
          value={formData.id}
          onChange={handleChange}
          placeholder="Enter unique identifier"
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="existing_name">Current Name</label>
        <input
          type="text"
          id="existing_name"
          name="existing_name"
          className="form-control"
          value={formData.existing_name}
          onChange={handleChange}
          placeholder="Enter current name (e.g. cust_acc_nbr)"
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="existing_description">Current Description</label>
        <textarea
          id="existing_description"
          name="existing_description"
          className="form-control"
          value={formData.existing_description}
          onChange={handleChange}
          placeholder="Enter current description"
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="example">Example (Optional)</label>
        <input
          type="text"
          id="example"
          name="example"
          className="form-control"
          value={formData.example}
          onChange={handleChange}
          placeholder="Enter example value"
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="cdm">CDM Category (Optional)</label>
        <input
          type="text"
          id="cdm"
          name="cdm"
          className="form-control"
          value={formData.cdm}
          onChange={handleChange}
          placeholder="Enter CDM category"
        />
      </div>
      
      <div className="form-actions">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading}
        >
          {loading ? (
            <span className="spinner"></span>
          ) : (
            <>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 22H15C20 22 22 20 22 15V9C22 4 20 2 15 2H9C4 2 2 4 2 9V15C2 20 4 22 9 22Z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M9 11.5C9 12.6 9.9 13.5 11 13.5H13C14.1 13.5 15 12.6 15 11.5C15 10.4 14.1 9.5 13 9.5H11C9.9 9.5 9 8.6 9 7.5C9 6.4 9.9 5.5 11 5.5H13C14.1 5.5 15 6.4 15 7.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M12 13.5V17" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M12 2V5.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              {submitLabel}
            </>
          )}
        </button>
      </div>
    </form>
  );
};

export default EnhancementForm;