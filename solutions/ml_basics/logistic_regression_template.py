"""
Logistic Regression Implementation Template
----------------------------------------
This template provides the structure for implementing logistic regression
from scratch using only NumPy.
"""

import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_: float = 0.1):
        """
        Initialize Logistic Regression model.
        
        Args:
            learning_rate: Step size for gradient descent
            n_iterations: Number of gradient descent steps
            regularization: Type of regularization ('l1', 'l2', or None)
            lambda_: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.history = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Implement the sigmoid activation function.
        
        Args:
            z: Input array
            
        Returns:
            Sigmoid of input
        """
        # Your code here
        pass
    
    def initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.
        
        Args:
            n_features: Number of input features
        """
        # Your code here
        pass
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with regularization.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            Total cost
        """
        # Your code here
        pass
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for gradient descent.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        # Your code here
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        # Your code here
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        # Your code here
        pass
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            threshold: Classification threshold
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        # Your code here
        pass
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot the decision boundary (only for 2D input).
        
        Args:
            X: Input features of shape (n_samples, 2)
            y: Target labels of shape (n_samples,)
        """
        # Your code here
        pass

def test_logistic_regression():
    """
    Test the LogisticRegression implementation.
    """
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    
    # Your test code here
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    model.plot_decision_boundary(X, y)
    plt.title('Decision Boundary')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_logistic_regression() 