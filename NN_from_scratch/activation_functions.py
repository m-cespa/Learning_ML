import numpy as np

class Linear:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
class Quadratic:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x**2
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 2*x
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        return 2*np.ones_like(x)
    
class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, np.array(x))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float64)
        
class ELU:
    def __call__(self, x: np.ndarray, alpha=1.) -> np.ndarray:
        x = np.clip(x, -100, 100)  # prevent overflow
        return np.where(x > 0, x, alpha * (np.expm1(x)))
        
    def derivative(self, x: np.ndarray, alpha=1.) -> np.ndarray:
        elu_x = self.__call__(x)
        return np.where(x > 0, 1, elu_x + alpha)
    
    def second_deriv(self, x: np.ndarray, alpha: float = 1.) -> np.ndarray:
        x = np.clip(x, -100, 100)
        return np.where(x > 0, 0.0, alpha * np.exp(x))

    def third_deriv(self, x: np.ndarray, alpha: float = 1.) -> np.ndarray:
        # for x > 0, zero; for x <= 0 same as second deriv
        x = np.clip(x, -100, 100)
        return np.where(x > 0, 0.0, alpha * np.exp(x))
    
class Tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x)**2
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        tanh_x = np.tanh(x)
        return -2 * tanh_x * (1.0 - tanh_x ** 2)
    
    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        tanh_x = np.tanh(x)
        return -2 * (1 - tanh_x**2) * (1 - 3 * tanh_x**2)
    
class Sin:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sin(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x)
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        return -np.sin(x)
    
    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        return -np.cos(x)
    
class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -100, 100)  # prevent overflow
        return 1 / (1 + np.exp(-np.array(x)))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def second_deriv(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.__call__(x)
        return sigmoid_x * (1.0 - sigmoid_x) * (1.0 - 2.0 * sigmoid_x)
    
    def third_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self.__call__(x)
        return s * (1 - s) * (1 - 6 * s + 6 * s**2)
