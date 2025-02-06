import numpy as np
from scipy.io import loadmat
import utils
import matplotlib.pyplot as plt
from scipy.optimize import minimize

INIT_EPSILON = 0.12
HIDDEN_LAYER_SIZE = 25
INPUT_LAYER_SIZE = 400
OUTPUT_LAYER_SIZE = 10

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g  = 1 / (1 + np.exp(-z))
    return g

def forward(theta1, theta2, X, y, lambda_):
    """
    Computes the forward propagation

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).
    
    Returns:
    --------
    h : array_like Shape (number of examples, output layer size)
        activation in the output layer for each example

    """
    m = X.shape[0]
    a1 = np.hstack([np.ones((m, 1)), X])
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack([np.ones((m,1)), a2])
    a3 = sigmoid(np.dot(a2, theta2.T))
    J = -(1/m) * np.sum(y * np.log(a3) + (1 - y) * np.log(1 - a3))
    regularization = lambda_ / (2*m) * (np.sum(theta1[:,1:] ** 2) + np.sum(theta2[:,1:] ** 2))
    J += regularization
    return a3, a2, a1, J

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 
    """
    a3,a2, a1, J = forward(theta1, theta2, X, y, lambda_)
    return J



def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """
    m = X.shape[0]
    a3,a2, a1, J, = forward(theta1, theta2, X, y, lambda_)
    delta3 = a3 - y
    delta2 = (np.dot(delta3, theta2) * (a2*(1-a2)))[:,1:]
    grad1 = np.dot(delta2.T,a1) / m
    grad2 = np.dot(delta3.T,a2) / m
    #regularization
    grad1[:,1:] += lambda_ / m * theta1[:,1:]
    grad2[:,1:] += lambda_ / m * theta2[:,1:]
    return (J, grad1, grad2)

def gradient_descent(X, y, theta1_in, theta2_in, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
    """
    theta1, theta2 = theta1_in, theta2_in
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
      J, dj_theta1, dj_theta2 = backprop(theta1, theta2,X,y,lambda_)
      theta1 = theta1 - alpha * dj_theta1
      theta2 = theta2 - alpha * dj_theta2
      J_history[i] = J
    return theta1, theta2, J_history

def run_gradient_descent(X,y, theta1_in, theta2_in, alpha, num_iters, lambda_=None):
    theta1, theta2, J_history = gradient_descent(X,y,theta1_in,theta2_in,1,num_iters,1)
    plt.figure()
    plt.title("J output at each iteration")
    plt.xlabel("Number of iterations")
    plt.ylabel("J = Cost Function")
    plt.plot(np.arange(num_iters), J_history)
    plt.savefig("Cost.png")
    return theta1, theta2

def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """
    m = X.shape[0]
    X1s = np.hstack([np.ones((m, 1)), X])
    hidden_layer = sigmoid(np.dot(X1s, theta1.T))
    hidden_layer1s = np.hstack([np.ones((m,1)), hidden_layer])
    out_layer = sigmoid(np.dot(hidden_layer1s, theta2.T))
    p = np.argmax(out_layer, axis = 1)
    return p

def backrpop_aux(theta, X, y, lambda_):
    theta1 = np.reshape(theta[:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],(HIDDEN_LAYER_SIZE, (INPUT_LAYER_SIZE + 1)))
    theta2 = np.reshape(theta[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],(OUTPUT_LAYER_SIZE, (HIDDEN_LAYER_SIZE + 1)))
    J, grad1, grad2 = backprop(theta1, theta2, X, y, lambda_)
    grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))
    return J, grad

def train(theta_in, X, y, lamda_):
    result = minimize(fun=backrpop_aux, x0=theta_in, args=(X,y,lamda_), method='TNC', jac=True, options={'maxfun' : 100})
    theta1 = np.reshape(result.x[:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],(HIDDEN_LAYER_SIZE, (INPUT_LAYER_SIZE + 1)))
    theta2 = np.reshape(result.x[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):],(OUTPUT_LAYER_SIZE, (HIDDEN_LAYER_SIZE + 1)))
    return theta1, theta2

def accuracy(theta1, theta2, X, y):
    p = predict(theta1, theta2, X)
    print(f"Accuracy: {np.sum(p == y) / p.shape[0] * 100}%")

