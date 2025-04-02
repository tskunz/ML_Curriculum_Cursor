"""
Linear Algebra Implementation Template
-----------------------------------
This template provides the structure for implementing basic linear algebra operations
from scratch using only NumPy arrays for storage.
"""

import numpy as np
from typing import List, Tuple, Optional

class LinearAlgebra:
    @staticmethod
    def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Implement matrix multiplication without using np.dot or @.
        
        Args:
            A: First matrix of shape (m, n)
            B: Second matrix of shape (n, p)
            
        Returns:
            Matrix product of shape (m, p)
            
        Raises:
            ValueError: If matrix dimensions don't match
        """
        # Your code here
        pass
    
    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the linear system Ax = b using Gaussian elimination.
        
        Args:
            A: Coefficient matrix of shape (n, n)
            b: Target vector of shape (n,)
            
        Returns:
            Solution vector x of shape (n,)
            
        Raises:
            ValueError: If matrix is singular or dimensions don't match
        """
        # Your code here
        pass
    
    @staticmethod
    def eigen_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find eigenvalues and eigenvectors using the power iteration method.
        
        Args:
            A: Square matrix of shape (n, n)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
            
        Raises:
            ValueError: If matrix is not square
        """
        # Your code here
        pass
    
    @staticmethod
    def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the Singular Value Decomposition using the power method.
        
        Args:
            A: Matrix of shape (m, n)
            
        Returns:
            Tuple of (U, S, V) where A = USV^T
            
        Note:
            This is a simplified version that returns only the principal components
        """
        # Your code here
        pass

def test_linear_algebra():
    """
    Test the LinearAlgebra class implementations.
    """
    # Test matrix multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    expected_product = np.array([[19, 22], [43, 50]])
    
    # Your test code here
    
    # Test linear system solver
    A = np.array([[2, 1], [1, 3]])
    b = np.array([4, 5])
    expected_solution = np.array([1, 1])
    
    # Your test code here
    
    # Test eigendecomposition
    A = np.array([[4, -1], [-1, 4]])
    
    # Your test code here
    
    # Test SVD
    A = np.array([[1, 0], [0, 1], [1, 1]])
    
    # Your test code here

if __name__ == "__main__":
    test_linear_algebra() 