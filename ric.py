import numpy as np

def initialize(dimentions):
    parameters={}
    C=len (dimentions)
    for c in range(1,C):
        parameters['W'+str(c)]=np.random.randn(dimentions[c],dimentions[c-1])
        parameters["b"+str(c)]=np.random.randn(dimentions[c],1)

    return parameters

def segmoid(Z):
        return 1/(1+np.exp(-Z))


def forward_propagation(X,paramaters):
    activation={"A0":X}
    C=len(paramaters)//2

    for c in range(1,C+1):
      Z=paramaters["W"+str(c)].dot(activation["A"+str(c-1)])+paramaters["b"+str(c)]
      activation["A"+str(c)]=segmoid(Z)

    return activation

def function_cout(A,y):
    return 1/len(y)*np.sum(-y*np.log(A)-(1-y)*np.log(1-A))

def gradients(A,X,y):
    dW=1/len(y)*np.dot(X.T,A-y)
    db=1/len(y)*np.sum(A-y)
    return(dW,db)




 