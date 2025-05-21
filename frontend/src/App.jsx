// src/App.jsx
import React, { useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import EnhancementForm from './components/EnhancementForm';
import EnhancementResult from './components/EnhancementResult';
import FileUpload from './components/FileUpload';
import Header from './components/Header';
import Footer from './components/Footer';
import { enhanceDataElement, validateDataElement, batchEnhanceElements } from './api/enhancementApi';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [activeTab, setActiveTab] = useState('single');

  const handleSingleEnhancement = async (dataElement) => {
    try {
      setLoading(true);
      const response = await enhanceDataElement({
        data_element: dataElement,
        max_iterations: 2
      });
      
      setResult(response.enhanced_data);
      setLoading(false);
      toast.success('Data element enhanced successfully!');
    } catch (error) {
      console.error('Enhancement error:', error);
      setLoading(false);
      toast.error(`Enhancement failed: ${error.message || 'Unknown error'}`);
    }
  };

  const handleValidation = async (dataElement) => {
    try {
      setLoading(true);
      const response = await validateDataElement(dataElement);
      setValidationResult(response);
      setLoading(false);
      toast.info(`Validation complete: ${response.quality_status}`);
    } catch (error) {
      console.error('Validation error:', error);
      setLoading(false);
      toast.error(`Validation failed: ${error.message || 'Unknown error'}`);
    }
  };

  const handleBatchEnhancement = async (dataElements) => {
    try {
      if (!dataElements.length) {
        toast.warning('No data elements to enhance');
        return;
      }
      
      setLoading(true);
      // Convert array of elements to array of enhancement requests
      const requests = dataElements.map(element => ({
        data_element: element,
        max_iterations: 2
      }));
      
      // Send for batch processing
      const requestIds = await batchEnhanceElements(requests);
      
      // For demo purposes, we'll simulate batch results
      // In a real app, you would poll the API for results using the requestIds
      const simulatedResults = dataElements.map((element, index) => ({
        original: element,
        enhanced: {
          id: element.id,
          existing_name: element.existing_name,
          existing_description: element.existing_description,
          enhanced_name: element.existing_name.toLowerCase().replace(/_/g, ' '),
          enhanced_description: `${element.existing_description.charAt(0).toUpperCase()}${element.existing_description.slice(1)}${element.existing_description.endsWith('.') ? '' : '.'}`,
          quality_status: Math.random() > 0.7 ? 'good' : Math.random() > 0.3 ? 'needs_improvement' : 'poor',
          confidence_score: 0.7 + (Math.random() * 0.3),
          enhancement_feedback: ["Converted to business-friendly format"]
        },
        requestId: requestIds[index]
      }));
      
      setBatchResults(simulatedResults);
      setActiveTab('batch');
      setLoading(false);
      toast.success(`Batch enhancement started for ${dataElements.length} elements`);
    } catch (error) {
      console.error('Batch enhancement error:', error);
      setLoading(false);
      toast.error(`Batch enhancement failed: ${error.message || 'Unknown error'}`);
    }
  };

  return (
    <div className="app">
      <Header />
      <main className="container">
        <div className="glass-panel main-content">
          <div className="tabs">
            <button 
              className={`tab ${activeTab === 'single' ? 'active' : ''}`} 
              onClick={() => setActiveTab('single')}
            >
              Single Enhancement
            </button>
            <button 
              className={`tab ${activeTab === 'validation' ? 'active' : ''}`} 
              onClick={() => setActiveTab('validation')}
            >
              Validation
            </button>
            <button 
              className={`tab ${activeTab === 'batch' ? 'active' : ''}`} 
              onClick={() => setActiveTab('batch')}
            >
              Batch Enhancement
            </button>
          </div>
          
          <div className="tab-content">
            {activeTab === 'single' && (
              <div className="single-enhancement">
                <div className="panel-grid">
                  <div className="glass-panel form-panel">
                    <h2>Enhance Data Element</h2>
                    <p>Improve your data element to meet ISO/IEC 11179 standards.</p>
                    <EnhancementForm onSubmit={handleSingleEnhancement} loading={loading} />
                  </div>
                  <div className="glass-panel result-panel">
                    <h2>Enhancement Result</h2>
                    <p>The enhanced data element will appear here after processing.</p>
                    {result && <EnhancementResult result={result} />}
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'validation' && (
              <div className="validation">
                <div className="panel-grid">
                  <div className="glass-panel form-panel">
                    <h2>Validate Data Element</h2>
                    <p>Check if your data element meets ISO/IEC 11179 standards.</p>
                    <EnhancementForm 
                      onSubmit={handleValidation} 
                      loading={loading} 
                      submitLabel="Validate Element"
                    />
                  </div>
                  <div className="glass-panel result-panel">
                    <h2>Validation Result</h2>
                    <p>The validation results will appear here after processing.</p>
                    {validationResult && (
                      <div className="validation-result">
                        <div className={`quality-badge quality-${validationResult.quality_status.toLowerCase()}`}>
                          {validationResult.quality_status.replace(/_/g, ' ')}
                        </div>
                        <div className="validation-details">
                          <h3>Feedback</h3>
                          <div className="validation-feedback">
                            {validationResult.feedback}
                          </div>
                          
                          {validationResult.suggested_improvements && validationResult.suggested_improvements.length > 0 && (
                            <>
                              <h3>Suggested Improvements</h3>
                              <ul className="suggested-improvements">
                                {validationResult.suggested_improvements.map((suggestion, idx) => (
                                  <li key={idx}>{suggestion}</li>
                                ))}
                              </ul>
                            </>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'batch' && (
              <div className="batch-enhancement">
                <div className="glass-panel">
                  <h2>Batch Enhancement</h2>
                  <p>Upload an Excel file to enhance multiple data elements at once.</p>
                  <FileUpload onUpload={handleBatchEnhancement} loading={loading} />
                  
                  {batchResults.length > 0 && (
                    <div className="batch-results">
                      <h3>Batch Results</h3>
                      <div className="batch-results-table">
                        <table>
                          <thead>
                            <tr>
                              <th>ID</th>
                              <th>Original Name</th>
                              <th>Enhanced Name</th>
                              <th>Quality</th>
                              <th>Confidence</th>
                              <th>Actions</th>
                            </tr>
                          </thead>
                          <tbody>
                            {batchResults.map((item, idx) => (
                              <tr key={idx}>
                                <td>{item.original.id}</td>
                                <td>{item.original.existing_name}</td>
                                <td>{item.enhanced.enhanced_name}</td>
                                <td>
                                  <span className={`quality-pill quality-${item.enhanced.quality_status}`}>
                                    {item.enhanced.quality_status.replace(/_/g, ' ')}
                                  </span>
                                </td>
                                <td>{Math.round(item.enhanced.confidence_score * 100)}%</td>
                                <td>
                                  <button 
                                    className="action-button"
                                    onClick={() => {
                                      setResult(item.enhanced);
                                      setActiveTab('single');
                                    }}
                                  >
                                    View
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      <Footer />
      <ToastContainer position="bottom-right" />
    </div>
  );
}

export default App;