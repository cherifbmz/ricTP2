import numpy as np

def initialize(dimentions):
    parameters={}
    C=len (dimentions)
    for c in range(1,C):
        parameters['W'+str(c)]=np.random.randn(dimentions[c],dimentions[c-1])
        parameters["b"+str(c)]=np.random.randn(dimentions[c],1)

    return parameters

def sigmoid(Z):
        return 1/(1+np.exp(-Z))


def forward_propagation(X,paramaters):
    activation={"A0":X}
    C=len(paramaters)//2

    for c in range(1,C+1):
      Z=paramaters["W"+str(c)].dot(activation["A"+str(c-1)])+paramaters["b"+str(c)]
      activation["A"+str(c)]=sigmoid(Z)

    return activation



def function_cout(A, y):
    m = y.shape[1]  
    return (-1/m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

def back_propagation(y,activations,paramaters):
    m=y.shape[1]
    C=len(paramaters)//2

    dZ=activations['A'+str(C)]-y
    gradients={}
    for c in reversed((range(1,C+1))):
        gradients["dW"+str(c)]=1/m*np.dot(dZ,activations["A"+str(c-1)].T)
        gradients["db"+str(c)]=1/m*np.sum(dZ,axis=1,keepdims=True )
        if c>1:
            dZ=np.dot(paramaters["W"+str(c)].T,dZ)*activations["A"+str(c-1)]*(1-activations["A"+str(c-1)])
   


    return gradients

def update_parameters(gradients,parameters,learning_rate):
    C=len(parameters)//2
     
    for c in range(1,C+1):
        parameters["W"+str(c)]=parameters["W"+str(c)]-learning_rate*gradients["dW"+str(c)]
        parameters["b"+str(c)]=parameters["b"+str(c)]-learning_rate*gradients["db"+str(c)] 

    return parameters

