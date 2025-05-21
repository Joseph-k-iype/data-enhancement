// src/setupTests.js
// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// src/App.test.js
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders Data Enhancement Platform header', () => {
  render(<App />);
  const headerElement = screen.getByText(/Data Enhancement Platform/i);
  expect(headerElement).toBeInTheDocument();
});

test('renders tab navigation', () => {
  render(<App />);
  const singleTabElement = screen.getByText(/Single Enhancement/i);
  const validationTabElement = screen.getByText(/Validation/i);
  const batchTabElement = screen.getByText(/Batch Enhancement/i);
  
  expect(singleTabElement).toBeInTheDocument();
  expect(validationTabElement).toBeInTheDocument();
  expect(batchTabElement).toBeInTheDocument();
});