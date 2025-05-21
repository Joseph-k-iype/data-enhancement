// src/api/enhancementApi.js
const API_BASE_URL = '/api/v1';

/**
 * Enhance a single data element
 * @param {Object} request - Enhancement request object containing data_element and max_iterations
 * @returns {Promise<Object>} - Enhancement response
 */
export const enhanceDataElement = async (request) => {
  try {
    const response = await fetch(`${API_BASE_URL}/enhance`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Enhancement failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Enhancement API error:', error);
    throw error;
  }
};

/**
 * Validate a data element against ISO/IEC 11179 standards
 * @param {Object} dataElement - Data element to validate
 * @returns {Promise<Object>} - Validation result
 */
export const validateDataElement = async (dataElement) => {
  try {
    const response = await fetch(`${API_BASE_URL}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(dataElement)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Validation failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Validation API error:', error);
    throw error;
  }
};

/**
 * Enhance multiple data elements in batch
 * @param {Array<Object>} requests - Array of enhancement requests
 * @returns {Promise<Array<string>>} - Array of request IDs
 */
export const batchEnhanceElements = async (requests) => {
  try {
    const response = await fetch(`${API_BASE_URL}/enhance/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requests)
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Batch enhancement failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Batch enhancement API error:', error);
    throw error;
  }
};

/**
 * Get the status and result of an enhancement job
 * @param {string} requestId - Enhancement request ID
 * @returns {Promise<Object>} - Enhancement status and result
 */
export const getEnhancementStatus = async (requestId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/enhance/${requestId}`, {
      method: 'GET'
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to get enhancement status');
    }

    return await response.json();
  } catch (error) {
    console.error('Enhancement status API error:', error);
    throw error;
  }
};