import numpy as np
import copy
import math
import matplotlib.pyplot as plt

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


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    pred = (X*w).sum(axis=1) + b
    f = sigmoid(pred)
    loss = (-1) * y * np.log(f) - (1 - y) * np.log(1-f) 
    total_cost = loss.sum() / X.shape[0]
    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """
    m = X.shape[0]
    pred = (X*w).sum(axis=1) + b
    f = sigmoid(pred)
    dj_db = (f - y).sum() / m
    dj_dw = (X.T * (f - y)).sum(axis=1) / m 
    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    m = X.shape[0]
    total_cost = compute_cost(X,y,w,b) + (w ** 2).sum() * lambda_ / (2*m)
    return total_cost 


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m = X.shape[0]
    dj_db, dj_dw = compute_gradient(X,y,w,b)
    dj_dw = dj_dw + w * lambda_ / m
    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    b = b_in
    w = w_in
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
      dj_db, dj_dw = gradient_function(X,y,w,b)
      b = b - alpha * dj_db
      w = w - alpha * dj_dw
      J_history[i] = cost_function(X,y,w,b)
      if i % 100 == 0:
         print(f"Iteration {i}")
    return w, b, J_history


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    pred = (X*w).sum(axis=1) + b
    p = sigmoid(pred) >= 0.5
    return p

def run_gradient_descent(X,y, reg = False):
  """"
   Runs the gradient descent algorithm, initializing the arguments required by the function
   and printing the coste per iteration

   Args
    X: (array_like Shape (m,n))   matriz of examples
    y: (array_like Shape (m,))    target value of each example
    reg: (boolean)                True if regularization must be applied
   Returns
    w: (array_like Shape (n,))  vector of weights of the line
    b: (scalar)                 displacement of the line
    J_history : (ndarray): Shape (num_iters,) J at each iteration,
      primarily for graphing later
  """
  num_iters = 10000
  w_in = np.zeros(X.shape[1])
  if reg is False:
    b_in = -8
    alpha = 0.001
    w,b,J_history = gradient_descent(X,y,w_in, b_in,compute_cost,compute_gradient,alpha,num_iters)
  else:
    b_in = 1
    alpha = 0.01
    lambda_ = 0.01
    w, b, J_history = gradient_descent(X,y,w_in,b_in, compute_cost_reg, compute_gradient_reg, alpha, num_iters, lambda_)

  title = "Cost2.png" if reg else "Cost1.png"

  plt.figure()
  plt.title("J output at each iteration")
  plt.xlabel("Number of iterations")
  plt.ylabel("J = Cost Function")
  plt.plot(np.arange(num_iters), J_history)
  plt.savefig(title)
  
  return w,b

def train(X,y,w_in,b, alpha, num_iters, lambda_):
    """
    Trains the parameters for a unique class
    """
    w,b, J = gradient_descent(X,y,w_in,b,compute_cost_reg, compute_gradient_reg,alpha, num_iters, lambda_)
    return w,b

