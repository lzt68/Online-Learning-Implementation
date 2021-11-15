# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:23:07 2021

@author: lzt68
"""
# following are functions that are related to the neural network
import numpy as np

def Relu(x):
    # Relu function
    # if x is a scalar, the return value would be a scalar
    # if x is a matrix, the return value would be a matrix
    return np.maximum(x,0)

def ReluDerivative(x):
    # the derivative of Relu function
    # if x is a scalar, the return value would be a scalar
    # if x is a matrix, the return value would be a matrix
#     return np.array([(lambda x: 1 if x > 0 else 0)(xi) for xi in x])
    if x > 0:
        return 1
    else:
        return 0

def NeuralNetwork(X, params, L, m):
    # X is the input, each column correspont to a context
    # params is a dictionay, each key corresponds to the weight in each layer
    # its keys are "w1" , "w2", ..., "w{:d}".format(L).
    # L is the number of layers
    # m is the number of neurals in each layer
    
    # the return value would be a dictionary, its key would be like "l0", "l1", "l2", ..., "l{:d}".format(L), 
    # each key represent the value of a layer, "l0" is X, the last layer is output
    
    X_layer = {}
    X_layer["x0"] = X
    for l in range(1, L):
        X_layer["x" + str(l)] = Relu(params["w" + str(l)].dot(X_layer["x" + str(l-1)]))
    X_layer["x" + str(L)] = np.sqrt(m) * params["w" + str(L)].dot(X_layer["x" + str(L-1)])
    return X_layer

# def GradientNeuralNetwork(X, params, L, m):
#     # this function calculate the gradient of each parameter in the neural network
#     # X is the context, a 1 dimension vecotr in R^d
#     # params is a dictionary that stores the paraemeters of neural network
#     # its keys are "w1" , "w2", ..., "w{:d}".format(L).
#     # L is the number of layers
#     # m is the number of neurals in each layer
    
#     # vectorize the function we used
#     myRelu = np.vectorize(Relu)
#     myReluDerivative = np.vectorize(ReluDerivative)
    
#     # we firstly calculate the value of each layer
#     X_layer = NeuralNetwork(X, params, L, m)
    
#     # then we calculate the gradient of "X_layer" and gradient of parameter
#     grad_X_layer = {}
#     grad_parameter = {}
    
#     grad_X_layer["x" + str(L)] = 1
#     grad_parameter["w" + str(L)] = np.sqrt(m) * np.expand_dims(X_layer["x" + str(L - 1)], axis = 0)
#     grad_X_layer["x" + str(L - 1)] = np.sqrt(m) * params["w" + str(L)][0, :]
#     for l in range(L - 1, 0, -1):
#         grad_parameter["w" + str(l)] = grad_X_layer["x" + str(l)] * myReluDerivative(X_layer["x" + str(l)])
#         # print(grad_X_layer["x" + str(l)].shape)
#         # print(myReluDerivative(X_layer["x" + str(l)]))
#         # print(grad_parameter["w" + str(l)].shape)
#         # print(X_layer["x" + str(l - 1)].shape)
#         grad_parameter["w" + str(l)] = np.matmul(np.expand_dims(grad_parameter["w" + str(l) ], axis=1),\
#                                                   np.expand_dims(X_layer["x" + str(l - 1)], axis = 0))

#         grad_X_layer["x" + str(l - 1)] = grad_X_layer["x" + str(l)] * myReluDerivative(X_layer["x" + str(l)])
#         # print(grad_X_layer["x" + str(l - 1)].shape)
#         # print(params["w" + str(l)].shape)
#         grad_X_layer["x" + str(l - 1)] = np.matmul(params["w" + str(l)].transpose(),\
#                                                    np.expand_dims(grad_X_layer["x" + str(l - 1)] , axis = 1))[:, 0]
        
#     return grad_parameter

def GradientNeuralNetwork(X, params, L, m):
    # this function calculate the gradient of each parameter in the neural network
    # X is the context, a 2-D matrix, each column represent a context
    # params is a dictionary that stores the paraemeters of neural network
    # its keys are "w1" , "w2", ..., "w{:d}".format(L)
    # L is the number of layers
    # m is the number of neurals in each layer
    
    # vectorize the function we used
    myRelu = np.vectorize(Relu)
    myReluDerivative = np.vectorize(ReluDerivative)
    
    if len(X.shape) == 1:
        # we firstly calculate the value of each layer
        X_layer = NeuralNetwork(X, params, L, m) # each value of X_layer would be a 1-D vector
        
        # then we calculate the gradient of "X_layer" and gradient of parameter
        grad_X_layer = {}
        grad_parameter = {}
        
        grad_X_layer["x" + str(L)] = 1
        grad_parameter["w" + str(L)] = np.sqrt(m) * np.expand_dims(X_layer["x" + str(L - 1)], axis = 0)
        grad_X_layer["x" + str(L - 1)] = np.sqrt(m) * params["w" + str(L)][0, :]
        for l in range(L - 1, 0, -1):
            grad_parameter["w" + str(l)] = grad_X_layer["x" + str(l)] * myReluDerivative(X_layer["x" + str(l)])
            # print(grad_X_layer["x" + str(l)].shape)
            # print(myReluDerivative(X_layer["x" + str(l)]))
            # print(grad_parameter["w" + str(l)].shape)
            # print(X_layer["x" + str(l - 1)].shape)
            grad_parameter["w" + str(l)] = np.matmul(np.expand_dims(grad_parameter["w" + str(l) ], axis=1),\
                                                      np.expand_dims(X_layer["x" + str(l - 1)], axis = 0))
    
            grad_X_layer["x" + str(l - 1)] = grad_X_layer["x" + str(l)] * myReluDerivative(X_layer["x" + str(l)])
            # print(grad_X_layer["x" + str(l - 1)].shape)
            # print(params["w" + str(l)].shape)
            grad_X_layer["x" + str(l - 1)] = np.matmul(params["w" + str(l)].transpose(),\
                                                        np.expand_dims(grad_X_layer["x" + str(l - 1)] , axis = 1))[:, 0]
    else:
        context_num = X.shape[1]
        X_layer = NeuralNetwork(X, params, L, m) # each value of X_layer would be a 2-D matrix
        # each column corresponds to  input X
        
        # then we calculate the gradient of "X_layer" and gradient of parameter
        grad_X_layer = {} # each value would be a 2-D matrix, correspond to X
        grad_parameter = {} # each value would be a 3-D matrix,
        # grad_parameter['w1'][0, :, :] would be the gradient of w1, a 2-D matrix, when the input is X[:, 0]
        
        grad_X_layer["x" + str(L)] = np.ones(shape = (1, context_num))
        # grad_X_layer["x" + str(L)] would be a 2-D matrix with, 1 * context_num, just because it is output
        
        grad_parameter["w" + str(L)] = np.sqrt(m) * np.expand_dims(X_layer["x" + str(L - 1)].transpose(), axis = 1)
        # print(grad_parameter["w" + str(L)])
        # print(X_layer)
        
        # grad_parameter["w" + str(L)] would be a 3-D matrix,
        # grad_parameter["w" + str(L)][0, :, :] correspond to the gradient of "wL" when the input is X[:, 0]
        # grad_parameter["w" + str(L)][0, :, :] would be a 2-D matrix, with the length of one dimension is 1 
        
        grad_X_layer["x" + str(L - 1)] = np.tile(np.expand_dims(np.sqrt(m) * params["w" + str(L)][0, :], axis = 1), (1, context_num))
        # grad_X_layer["x" + str(L - 1)] would be a 2-D matrix,
        # grad_X_layer["x" + str(L - 1)][:, 0] correspond to the gradient of "x_L-1" when the input is X[:, 0]
        # np.tile(np.expand_dims(np.array([1, 2, 3]), axis = 1), (1, 3))
        for l in range(L - 1, 0, -1):
            temp_grad_this_layer = grad_X_layer["x" + str(l)] * myReluDerivative(X_layer["x" + str(l)])
            # grad_X_layer["x" + str(l)] and myReluDerivative(X_layer["x" + str(l)]) are both 2-D matrix
            # the multiplication here is element-wised
            # temp_grad_this_layer is a temp variable
            
            temp_grad_this_layer = np.expand_dims(temp_grad_this_layer.transpose(), axis = 1).transpose([0, 2, 1])
            # temp_grad_this_layer would be a 3-D matrix, temp_grad_this_layer[0, :, :] would be a 2-D matrix
            # whose width is 1
            # temp_grad_this_layer[0, :, :] corresponds to the gradient of X layer when input is X[:, 0]
            # np.expand_dims(np.array([[1, 2, 3], [0, 0, 1]]).transpose(), axis = 1).transpose([0, 2, 1]) = 
            # np.array([ [[1], [0]], [[2], [0]], [[3], [1]] ])
            
            temp_X_last_layer = np.expand_dims(X_layer["x" + str(l - 1)].transpose(), axis = 1)
            # temp_X_last_layer would be a 3-D matrix
            # temp_X_last_layer[0, :, :] would be a 2-D matrix whose height is 1
            # temp_X_last_layer[0, :, :] is the X_layer value corresponds to X[:, 0]
            # np.expand_dims(np.array([[1, 2, 3], [0, 0, 1]]).transpose(), axis = 1) = 
            # np.array([ [[1,0]], [[2, 0]], [[3, 1]] ])
            
            grad_parameter["w" + str(l)] = np.matmul(temp_grad_this_layer, temp_X_last_layer)
            # temp_grad_this_layer and temp_X_last_layer share the same length on axis = 0
            # grad_parameter["w" + str(l)] would be a 3-D matrix, 
            # grad_parameter["w" + str(l)][0, :, :] corresponds to the gradient of wl when input is X[:, 0]
            # np.matmul(np.array([ [[1], [0]], [[2], [0]], [[3], [1]] ]), np.array([ [[1,0]], [[2, 0]], [[3, 1]] ])) =
            # np.array([ [[1,0],[0,0]], [[4,0],[0,0]], [[9,3],[3,1]] ])
            
    
            grad_X_layer["x" + str(l - 1)] = np.matmul(params["w" + str(l)].transpose(), temp_grad_this_layer)
            # grad_X_layer["x" + str(l - 1)] would be a 3-D matrix
            # grad_X_layer[0, : , :] would be the gradient of x_l-1, when the input is X[:, 0]
            # np.matmul(np.array([ [[1,0],[0,0]], [[4,0],[0,0]], [[9,3],[3,1]] ]), np.array([ [[1], [0]], [[2], [0]], [[3], [1]] ])) =
            # np.array([ [[1],[0]], [[4],[0]], [[30],[10]] ])
            grad_X_layer["x" + str(l - 1)] = grad_X_layer["x" + str(l - 1)][:, :, 0].transpose()
            # we deduct the shape of x_l-1
            # np.array([ [[1],[0]], [[4],[0]], [[30],[10]] ])[:, :, 0].transpose() = 
            # np.array([ [1, 4, 30], [0, 0, 10] ])
            
            
    return grad_parameter

def FlattenDict(para_dict, L):
    # accroding to the paper, the function f should be a vector in R^p
    # but what we got in GradientNeuralNetwork is a dictionary,
    # we would use this function to convert the dictinary into a vector
    # para_dict is the gradient stored in a dictionary
    # L is the total number of layers
    
    # the return value is the flattern vector = [vec(w1)^T, vec(w2)^T, ..., vec(w_L)^T]^T
    
    # we firstly generate the order of all the parameter
    para_order = ["w" + str(l) for l in range(1, L + 1)] #e.g. para_order = ["w1", "w2", "w3"]
    # then we can combine all the 1 d arrays together
    para = np.concatenate([para_dict[para_name].flatten() for para_name in para_order])
    return para

def LossFunction(X, params, L, m, r, theta_0, lambda_):
    # this function calculate the Loss function
    # X is the matrix of observed contexts, each column represent a context
    # X is required to be a 2-D matrix here, even sometimes it may only contain one column
    # params is a dictionary that stores the paraemeters of neural network
    # its keys are "w1" , "w2", ..., "w{:d}".format(L).
    # L is the number of layers
    # m is the number of neurals in each layer
    # r is the reward in each round
    # r is required to be a vector here, even sometimes it may only contain one column
    # theta_0 is the initalized value of parameters, restored as a disctionary
    
    X_layer = NeuralNetwork(X, params, L, m) # each value in X_layer would be a 2-D matrix
    
    predicted = X_layer["x" + str(L)][0, :]
    term1 = np.sum(np.square(predicted - r)) / 2
    theta_0 = FlattenDict(theta_0, L)
    params = FlattenDict(params, L)
    term2 = m * lambda_ * np.sum(np.square(params - theta_0)) / 2
    
    return term1 + term2
    
def GradientLossFunction(X, params, L, m, r, theta_0, lambda_):
    # this function calculate the gradient of lossfunction
    # X is the matrix of observed contexts, each column represent a context
    # X is required to be a 2-D matrix here, even sometimes it may only contain one column
    # params is a dictionary that stores the paraemeters of neural network
    # its keys are "w1" , "w2", ..., "w{:d}".format(L).
    # L is the number of layers
    # m is the number of neurals in each layer
    # r is the reward in each round
    # r is required to be a vector here, even sometimes it may only contain one column
    # theta_0 is the initalized value of parameters, restored as a dictionary
    
    
    '''
    old version
    #we would repeatedly call GradientNeuralNetwork() to calculate the gradients here
    # # firstly, we calculate the shape of X and r
    # context_num = len(r)
    
    # # secondly, we calculate the value of each layer
    # X_layer = NeuralNetwork(X, params, L, m) # each value in X_layer would be a 2-D matrix
    
    # # secondly, we repeatedly call GradientNeuralNetwork() to calculate the gradient of regression part
    # grad_loss = {}# apply for space
    # for key in params.keys():
    #     grad_loss[key] = np.zeros(params[key].shape)
    
    # for ii in range(1, context_num + 1):
    #     new_term = GradientNeuralNetwork(X[:, ii - 1], params, L, m)
    #     for key in grad_loss.keys():
    #         grad_loss[key] = grad_loss[key] + new_term[key] * (X_layer["x" + str(L)][0, ii - 1] - r[ii - 1])
    
    # thirdly, we calculate the gradient of regularization
    # for key in grad_loss.keys():
    #     grad_loss[key] = grad_loss[key] + m * lambda_ * (params[key] - theta_0[key])
    '''
    
    # firstly, we calculate the value of each layer
    X_layer = NeuralNetwork(X, params, L, m) # each value in X_layer would be a 2-D matrix
    
    # secondly, we call GradientNeuralNetwork() to calculate the gradient of regression part
    grad_loss = {}# apply for space
    grad_param = GradientNeuralNetwork(X, params, L, m)
    for key in params.keys():
        shape_of_weight = grad_param[key][0, :, :].shape
        temp_gap = np.stack([value * np.ones(shape = shape_of_weight) for value in X_layer["x" + str(L)][0, :] - r[:]], axis = 0)
        # np.stack([value * np.ones(shape = (2, 2)) for value in np.array([1, 2, 3])], axis = 1) = 
        # np.array([ [[1., 1.],[1., 1.]], [ [2., 2.],[2., 2.] ], [ [3., 3.],[3., 3.]] ])
        grad_loss[key] = grad_param[key] * temp_gap
        grad_loss[key] = np.sum(grad_loss[key], axis = 0)
        # np.sum(np.array([ [[1., 1.],[1., 1.]], [ [2., 2.],[2., 2.] ], [ [3., 3.],[3., 3.]] ]), axis = 0) = 
        # np.array([[6, 6], [6, 6]])
    
    # thirdly, we calculate the gradient of regularization
    for key in grad_loss.keys():
        grad_loss[key] = grad_loss[key] + m * lambda_ * (params[key] - theta_0[key])
    
    return grad_loss

# X = np.array([[1, 1], [2, 3]])
# lambda_ = 0
# theta_0 = {"w1": np.zeros((4, 2)),
#            "w2": np.zeros((1,4))}
# r = np.array([0, 0])
# L = 2
# m = 4
# params = {"w1": np.array([[1, 2], [-3, 2], [0, -1], [3, 3]]),
#           "w2": np.array([[1, 2, 3, -1]])}
# grad_loss = GradientLossFunction(X, params, L, m, r, theta_0, lambda_)
# X_layer = NeuralNetwork(X, params, L, m)
# print(X_layer)
# print(grad_loss)

# grad_parameter = GradientNeuralNetwork(X[:, 0], params, L, m)
# print(grad_parameter)
# grad_parameter = GradientNeuralNetwork(X[:, 1], params, L, m)
# print(grad_parameter)