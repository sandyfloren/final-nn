# TODO: import dependencies and write unit tests below
from neural import NeuralNetwork, sample_seqs, one_hot_encode_seqs
from sklearn import metrics
import numpy as np
import pytest

# Create a NeuralNetwork instance with multiple layers and activation functions
nn_arch = [
    {
        'input_dim': 64,
        'output_dim': 16, 
        'activation': 'relu'
    },
    {
        'input_dim': 16, 
        'output_dim': 64, 
        'activation': 'relu'
    },
    {
        'input_dim': 64,
        'output_dim': 10, 
        'activation': 'sigmoid'
    }
]

lr = 0.01
seed = 462
batch_size = 4
epochs = 100
loss_function = 'binary_cross_entropy'

net = NeuralNetwork(nn_arch,
                    lr,
                    seed,
                    batch_size,
                    epochs,
                    loss_function)


def test_single_forward():

    # Test many possible dimensions
    for n in range(1, 100):
        for m in range(1, 100):

            W_curr = np.random.normal(size=(n, m))
            b_curr = np.random.normal(size=(n))
            A_prev = np.random.normal(size=(m))

            A_curr_relu, Z_curr_relu = net._single_forward(W_curr, b_curr, A_prev, 'relu')
            A_curr_sig, Z_curr_sig = net._single_forward(W_curr, b_curr, A_prev, 'sigmoid')

            assert np.all(Z_curr_relu == W_curr @ A_prev + b_curr)
            assert np.all(A_curr_relu == np.maximum(Z_curr_relu, 0))
            assert np.all(Z_curr_sig == Z_curr_relu)
            assert np.all(A_curr_sig == 1 / (1 + np.exp(-Z_curr_relu)))


def test_forward():

    X = np.random.random((4, 64))
    output, cache = net.forward(X)

    # Check output shape
    assert output.shape == (10, 4)

    # Check that cache was built correctly
    assert set(cache.keys()) == {'A0', 'Z1', 'A1', 'Z2', 'A2', 'Z3', 'A3'}
    

def test_single_backprop():
    # Test many possible dimensions
    for n in range(1, 100):
        for m in range(1, 100):

            W_curr = np.random.random(size=(n, m))
            b_curr = np.random.random(size=(n, 1))
            Z_curr = np.random.random(size=(n, 1))
            A_prev = np.random.random(size=(m, 1))
            dA_curr = np.random.random(size=(n, 1))

            dZ_relu = dA_curr * (Z_curr > 0).astype(float)
            sig_Z = 1 / (1 + np.exp(-Z_curr))
            dZ_sig = dA_curr * sig_Z * (1 - sig_Z)

            dA_prev_relu, dW_curr_relu, db_curr_relu = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'relu')
            dA_prev_sig, dW_curr_sig, db_curr_sig = net._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, 'sigmoid')

            assert np.all(dA_prev_relu == W_curr.T @ dZ_relu)
            assert np.all(dW_curr_relu == dZ_relu @ A_prev.T)
            assert np.all(db_curr_relu == np.sum(dZ_relu, axis=1, keepdims=True))

            assert np.all(dA_prev_sig == W_curr.T @ dZ_sig)
            assert np.all(dW_curr_sig == dZ_sig @ A_prev.T)
            assert np.all(db_curr_sig == np.sum(dZ_sig, axis=1, keepdims=True))
            

def test_predict():

    X = np.random.random((4, 64))
    output, cache = net.forward(X)
    preds = net.predict(X)
    
    # Ensure that predict output is the same as forward output
    assert np.all(output == preds)

def test_binary_cross_entropy():
    
    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    # Compare to scikit-learn cross-entropy loss
    assert np.isclose(net._binary_cross_entropy(y, y_hat), metrics.log_loss(y, y_hat))

def test_binary_cross_entropy_backprop():

    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    epsilon = 1e-12
    dA = (1/batch_size) * (-y/(y_hat + epsilon) + (1-y)/(1-y_hat + epsilon))
    assert np.all(net._binary_cross_entropy_backprop(y, y_hat) == dA)

def test_mean_squared_error():

    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    # Compare to scikit-learn mean squared error
    assert np.isclose(net._mean_squared_error(y, y_hat), metrics.mean_squared_error(y, y_hat))

def test_mean_squared_error_backprop():
    
    y_hat = np.random.uniform(low=0, high=1, size=batch_size)
    y = np.random.randint(low=0, high=2, size=batch_size)

    dA = (1/batch_size) * -2*(y - y_hat)
    assert np.all(net._mean_squared_error_backprop(y, y_hat) == dA)

def test_sample_seqs():

    with pytest.raises(ValueError):
        sample_seqs([], [], 62)
    with pytest.raises(ValueError):
        sample_seqs(['test'], [False], 62)
    with pytest.raises(ValueError):
        sample_seqs(['test', 'test'], [True, True], 62)

    # Test multiple lengths
    for length in range(2, 100):

        seqs = ['test' for _ in range(0, length)]
        labels = list(np.random.choice(a=[True, False], size=length))
        
        # Don't test if there aren't examples from both classes
        if len(np.unique(labels)) == 1:
            continue

        sampled_seqs, sampled_labels = sample_seqs(seqs, labels, 62)

        assert len(sampled_seqs) == len(sampled_labels)
        assert len(np.where(sampled_labels)[0]) == len(sampled_labels) / 2

def test_one_hot_encode_seqs():

    # Test multiple lengths and numbers of sequences
    for length in range(0, 50):
        for n in range(0, 50):

            seqs = [''.join(np.random.choice(a=['A', 'C', 'T', 'G'], size=length)) for _ in range(n)]

            encodings = one_hot_encode_seqs(seqs)

            # Check length
            assert len(encodings) == length * n * 4

            # Check values
            if length == 0 or n == 0:
                assert encodings.size == 0
            else:
                assert len(np.where(encodings == 1)[0]) == length * n
                assert len(np.where(encodings == 0)[0]) == length * n * 3