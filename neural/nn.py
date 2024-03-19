# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

        # Create dictionaries for activation functions and their derivatives
        self._activation_dict = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'softmax': self._softmax
            }
        self._activation_backprop_dict = {
            'relu': self._relu_backprop,
            'sigmoid': self._sigmoid_backprop,
            'softmax': self._softmax_backprop
            }
        
        # Create dictionaries for loss functions and their derivatives
        self._loss_func_dict = {
            'binary_cross_entropy': self._binary_cross_entropy,
            'mean_squared_error': self._mean_squared_error
            }
        self._loss_func_backprop_dict = {
            'binary_cross_entropy': self._binary_cross_entropy_backprop,
            'mean_squared_error': self._mean_squared_error_backprop
        }
        
        self.n_layers = len(nn_arch)


    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Compute Z and A
        f = self._activation_dict[activation]
        Z_curr = W_curr @ A_prev + b_curr
        A_curr = f(Z_curr)
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialize cache
        cache = {}

        # Use X as inputs to first layer 
        A_prev = X.T
        cache['A0'] = X.T
        for i in range(self.n_layers):
            layer_idx = i + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = self.arch[i]['activation']

            # Compute activations for this layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Store Z and A matrices in cache
            cache['A' + str(layer_idx)] = A_curr
            cache['Z' + str(layer_idx)] = Z_curr

            A_prev = A_curr

        output = A_prev

        return output, cache
            

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Compute partial derivatives with respect to W, b, and A
        activation_backprop = self._activation_backprop_dict[activation_curr]
        dZ = activation_backprop(dA_curr, Z_curr)
        dW_curr = dZ @ A_prev.T
        db_curr = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = W_curr.T @ dZ

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        loss_func_backprop = self._loss_func_backprop_dict[self._loss_func]

        grad_dict = {}
        dA_curr = loss_func_backprop(y, y_hat)
        for i in range(self.n_layers-1, -1, -1):
            layer_idx = i + 1
            grad_dict['dA' + str(layer_idx)] = dA_curr
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            Z_curr = cache['Z' + str(layer_idx)]
            A_prev = cache['A' + str(i)]
            activation_curr = self.arch[i]['activation']

            # Compute activations for this layer
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr)

            # Store derivatives in cache
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

            dA_curr = dA_prev

        grad_dict['dA1'] = dA_curr

        return grad_dict
    
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Update model weights and biases for all layers
        for i in range(self.n_layers):
            layer_idx = i + 1
            self._param_dict['W' + str(layer_idx)] = self._param_dict['W' + str(layer_idx)] - self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] = self._param_dict['b' + str(layer_idx)] - self._lr * np.mean(grad_dict['db' + str(layer_idx)])

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        loss_func = self._loss_func_dict[self._loss_func]
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        # Loop over epochs
        for epoch in range(self._epochs):
            epoch_train_loss = 0
            epoch_val_loss = 0
            train_indices = np.arange(len(X_train))
            val_indices = np.arange(len(X_val))
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]
            X_val = X_val[val_indices]
            y_val = y_val[val_indices]

            # Train
            for idx in range(0, len(X_train), self._batch_size):
                next_idx = idx + self._batch_size
                X_train_batch = X_train[idx:next_idx]
                y_train_batch = y_train[idx:next_idx]

                # Forward pass
                y_hat, cache = self.forward(X_train_batch)
                train_loss_batch = loss_func(y_train_batch.T, y_hat)

                # Backward pass
                grad_dict = self.backprop(y_train_batch.T, y_hat, cache)
                self._update_params(grad_dict)

                # Log loss
                epoch_train_loss += train_loss_batch

            per_epoch_loss_train.append(epoch_train_loss)

            if X_val is not None and y_val is not None:
                # Validate
                for idx in range(0, len(X_val), self._batch_size):
                    next_idx = idx + self._batch_size
                    X_val_batch = X_val[idx:next_idx]
                    y_val_batch = y_val[idx:next_idx]

                    # Make predictions on validation set
                    y_hat = self.predict(X_val_batch)
                    val_loss_batch = loss_func(y_val_batch.T, y_hat)
                    epoch_val_loss += val_loss_batch

                # Log loss
                per_epoch_loss_val.append(epoch_val_loss)
        
        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # Use X as inputs to first layer 
        A_prev = X.T
        for i in range(self.n_layers):
            layer_idx = i + 1
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]
            activation = self.arch[i]['activation']

            # Compute activations for this layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            A_prev = A_curr

        y_hat = A_prev

        return y_hat
    
    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """ 
        sig_Z = self._sigmoid(Z)
        d_sigmoid = dA * sig_Z * (1 - sig_Z)
        return d_sigmoid


    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        return dA * (Z > 0).astype(float)
    
    def _softmax(self, Z: ArrayLike) -> ArrayLike:
        """
        Softmax activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        softmax = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)))
        return softmax

    def _softmax_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        Softmax derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        softmax = np.reshape(self._softmax(Z), (1, -1))
        d_softmax = (softmax * np.identity(softmax.size) - softmax.T @ softmax)
        return dA @ np.sum(d_softmax, axis=1)

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        epsilon = 1e-12
        loss = -(1/self._batch_size) * np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        epsilon = 1e-12
        dA = (1/self._batch_size) * (-y/(y_hat + epsilon) + (1-y)/(1-y_hat + epsilon))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = (1/self._batch_size) * np.sum((y - y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA = (1/self._batch_size) * -2*(y - y_hat)
        return dA