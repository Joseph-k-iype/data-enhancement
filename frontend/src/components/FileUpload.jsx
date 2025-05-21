// src/components/FileUpload.jsx
import React, { useState, useRef } from 'react';
import * as XLSX from 'xlsx';

const FileUpload = ({ onUpload, loading }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState([]);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setError('');
    
    if (!selectedFile) {
      setFile(null);
      setPreview([]);
      return;
    }
    
    // Check if file is Excel
    const validExts = ['.xlsx', '.xls', '.csv'];
    const ext = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    
    if (!validExts.includes(ext)) {
      setError('Please select an Excel or CSV file');
      setFile(null);
      setPreview([]);
      return;
    }
    
    setFile(selectedFile);
    parseExcel(selectedFile);
  };
  
  const parseExcel = (file) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { 
          type: 'array',
          cellDates: true,
          cellStyles: true
        });
        
        // Get first sheet
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Convert to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        if (jsonData.length < 2) {
          setError('File must contain at least one data row and a header row');
          setPreview([]);
          return;
        }
        
        // Extract headers (first row)
        const headers = jsonData[0];
        
        // Map expected columns
        const columnMap = {
          id: findColumnIndex(headers, ['id', 'element_id', 'element id', 'data_element_id']),
          name: findColumnIndex(headers, ['name', 'element_name', 'current_name', 'existing_name']),
          description: findColumnIndex(headers, ['description', 'element_description', 'current_description', 'existing_description']),
          example: findColumnIndex(headers, ['example', 'sample', 'example_value']),
          cdm: findColumnIndex(headers, ['cdm', 'cdm_category', 'category'])
        };
        
        // Check if required columns exist
        if (columnMap.id === -1 || columnMap.name === -1 || columnMap.description === -1) {
          setError('File must contain columns for ID, Name, and Description');
          setPreview([]);
          return;
        }
        
        // Process data rows
        const dataElements = [];
        
        for (let i = 1; i < jsonData.length; i++) {
          const row = jsonData[i];
          
          // Skip empty rows
          if (!row.length) continue;
          
          const element = {
            id: row[columnMap.id]?.toString() || `DE-${i}`,
            existing_name: row[columnMap.name]?.toString() || '',
            existing_description: row[columnMap.description]?.toString() || '',
            example: columnMap.example !== -1 ? row[columnMap.example]?.toString() || '' : '',
            cdm: columnMap.cdm !== -1 ? row[columnMap.cdm]?.toString() || '' : '',
            processes: []
          };
          
          // Skip rows without name or description
          if (!element.existing_name || !element.existing_description) continue;
          
          dataElements.push(element);
        }
        
        if (dataElements.length === 0) {
          setError('No valid data elements found in file');
          setPreview([]);
          return;
        }
        
        // Set preview (limit to 5)
        setPreview(dataElements.slice(0, 5));
      } catch (error) {
        console.error('Error parsing Excel file:', error);
        setError('Failed to parse file: ' + error.message);
        setPreview([]);
      }
    };
    
    reader.onerror = () => {
      setError('Failed to read file');
      setPreview([]);
    };
    
    reader.readAsArrayBuffer(file);
  };
  
  const findColumnIndex = (headers, possibleNames) => {
    for (const name of possibleNames) {
      const index = headers.findIndex(h => 
        h && h.toString().toLowerCase().replace(/[^a-z0-9]/g, '') === name.toLowerCase().replace(/[^a-z0-9]/g, '')
      );
      if (index !== -1) return index;
    }
    return -1;
  };
  
  const handleSubmit = async () => {
    if (!file) return;
    
    try {
      const reader = new FileReader();
      
      reader.onload = async (e) => {
        try {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          
          // Get first sheet
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          
          // Convert to JSON
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
          
          // Extract headers (first row)
          const headers = jsonData[0];
          
          // Map expected columns
          const columnMap = {
            id: findColumnIndex(headers, ['id', 'element_id', 'element id', 'data_element_id']),
            name: findColumnIndex(headers, ['name', 'element_name', 'current_name', 'existing_name']),
            description: findColumnIndex(headers, ['description', 'element_description', 'current_description', 'existing_description']),
            example: findColumnIndex(headers, ['example', 'sample', 'example_value']),
            cdm: findColumnIndex(headers, ['cdm', 'cdm_category', 'category'])
          };
          
          // Process data rows
          const dataElements = [];
          
          for (let i = 1; i < jsonData.length; i++) {
            const row = jsonData[i];
            
            // Skip empty rows
            if (!row || !row.length) continue;
            
            const element = {
              id: row[columnMap.id]?.toString() || `DE-${i}`,
              existing_name: row[columnMap.name]?.toString() || '',
              existing_description: row[columnMap.description]?.toString() || '',
              example: columnMap.example !== -1 && row[columnMap.example] 
                ? row[columnMap.example].toString() 
                : '',
              cdm: columnMap.cdm !== -1 && row[columnMap.cdm] 
                ? row[columnMap.cdm].toString() 
                : '',
              processes: []
            };
            
            // Skip rows without name or description
            if (!element.existing_name || !element.existing_description) continue;
            
            dataElements.push(element);
          }
          
          if (dataElements.length === 0) {
            setError('No valid data elements found in file');
            return;
          }
          
          // Call the onUpload callback with the parsed data elements
          await onUpload(dataElements);
          
          // Reset file selection
          // setFile(null);
          // fileInputRef.current.value = null;
        } catch (error) {
          console.error('Error processing Excel file:', error);
          setError('Failed to process file: ' + error.message);
        }
      };
      
      reader.onerror = () => {
        setError('Failed to read file');
      };
      
      reader.readAsArrayBuffer(file);
    } catch (error) {
      console.error('Error processing file:', error);
      setError('Failed to process file: ' + error.message);
    }
  };

  return (
    <div className="file-upload">
      <div className="upload-container">
        <input
          type="file"
          id="excel-upload"
          onChange={handleFileChange}
          accept=".xlsx,.xls,.csv"
          className="file-input"
          ref={fileInputRef}
        />
        <label htmlFor="excel-upload" className="upload-label">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 22H15C20 22 22 20 22 15V9C22 4 20 2 15 2H9C4 2 2 4 2 9V15C2 20 4 22 9 22Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M9 10C10.1046 10 11 9.10457 11 8C11 6.89543 10.1046 6 9 6C7.89543 6 7 6.89543 7 8C7 9.10457 7.89543 10 9 10Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M2.67 18.95L7.6 15.64C8.39 15.11 9.53 15.17 10.24 15.78L10.57 16.07C11.35 16.74 12.61 16.74 13.39 16.07L17.55 12.5C18.33 11.83 19.59 11.83 20.37 12.5L22 13.9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <span>
            {file ? file.name : 'Choose Excel or CSV file'}
          </span>
        </label>
      </div>
      
      {error && (
        <div className="upload-error">{error}</div>
      )}
      
      {preview.length > 0 && (
        <div className="file-preview">
          <h3>Preview ({preview.length} of {Math.min(5, preview.length)} elements)</h3>
          <table className="preview-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {preview.map((element, idx) => (
                <tr key={idx}>
                  <td>{element.id}</td>
                  <td>{element.existing_name}</td>
                  <td>{element.existing_description}</td>
                </tr>
              ))}
            </tbody>
          </table>
          
          <div className="upload-actions">
            <button 
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                <>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 10.5L11 12.5L15.5 8" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2C6.5 2 2 6.5 2 12C2 17.5 6.5 22 12 22Z" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Process {preview.length} Elements
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;