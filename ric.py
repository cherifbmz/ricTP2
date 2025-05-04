from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os 
import pickle

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
    return (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

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

def predict(X, parameters):
    activations = forward_propagation(X, parameters)
    C = len(parameters) // 2
    return (activations['A' + str(C)] > 0.5).astype(int)

def train_mlp(X, y, hidden_layers, learning_rate=0.1, n_iter=1000):
    np.random.seed(0)
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    parameters = initialize(dimensions)
    train_loss = []
    train_acc = []
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parameters)
        gradients = back_propagation(y, activations, parameters)
        parameters = update_parameters(gradients, parameters, learning_rate)
        if i % 10 == 0:
            C = len(parameters) // 2
            train_loss.append(function_cout(activations['A' + str(C)], y))
            y_pred = predict(X, parameters)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].legend()
    ax[1].plot(train_acc, label='Train Accuracy')
    ax[1].legend()
    plt.show()
    return parameters


def save_parameters(parameters, filename='model_parameters.pkl'):
    
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)
    print("Model parameters saved in ",filename)

def load_parameters(filename='model_parameters.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            parameters = pickle.load(f)
        print("Model parameters loaded from ",filename)
        return parameters
    else:
        print(filename," not found.")
        return None

def train_or_load_model(X, y, hidden_layers, filename='model_parameters.pkl', 
                        force_train=False, learning_rate=0.1, n_iter=1000):
    if not force_train and os.path.exists(filename):
        print("Loading existing model from ",filename)
        return load_parameters(filename)
    else:
        print("Training new model...")
        parameters = train_mlp(X, y, hidden_layers, learning_rate, n_iter)
        save_parameters(parameters, filename)
        return parameters

#X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
#y = np.array([[0, 1, 1, 0]])
#parameters = train_mlp(X, y, hidden_layers=(4, 4), learning_rate=0.1, n_iter=5000)
#y_pred= predict(X, parameters)
#print("Predictions:", y_pred.flatten())
#print("Actual:", y.flatten())
#print("Accuracy:", accuracy_score(y.flatten(), y_pred.flatten()))

data=np.loadtxt("recTp/data.txt")
print(data)

X=data[:,:-1].T
print(X)
Y=data[:,-1].reshape(1, -1)

print(Y)

X_training,Xtest,Y_training,Ytest=train_test_split(X.T,Y.T,test_size=0.2,random_state=1)
X_training=X_training.T
Y_training=Y_training.T
Xtest=Xtest.T
Ytest=Ytest.T

parameters = train_or_load_model(
        X_training, 
        Y_training, 
        hidden_layers=(16, 16, 16),
        filename='recTp/original_model.pkl',
        force_train=False,  
        learning_rate=1, 
        n_iter=10000
    )
ytraining_pred= predict(X_training, parameters)
ytest_pred=predict(Xtest,parameters)

print("Predictions on the training set befor opitmization :", ytraining_pred.flatten())
print("Actual:", Y_training.flatten())
print("Accuracy:", accuracy_score(Y_training.flatten(), ytraining_pred.flatten()))

print("Predictions on the test set before optimization :", ytest_pred.flatten())
print("Actual:", Ytest.flatten())
print("Accuracy:", accuracy_score(Ytest.flatten(), ytest_pred.flatten()))

def plot_3d_predictions(Xtest, Ytest, ytest_pred):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    Xtest = Xtest.T  
    Ytest = Ytest.flatten()
    ytest_pred = ytest_pred.flatten()

    for i in range(Xtest.shape[0]):
        color = 'green' if ytest_pred[i] == 0 else 'red'
        ax.scatter(Xtest[i, 0], Xtest[i, 1], Xtest[i, 2], c=color, s=5)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('3D Predictions: Green=Correct, Red=Incorrect')
    plt.show()

plot_3d_predictions(Xtest, Ytest, ytest_pred)


x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
r = np.sqrt(x ** 2 + y ** 2)

X = np.vstack((r, z)).T
X = X.T

X_training,Xtest,Y_training,Ytest=train_test_split(X.T,Y.T,test_size=0.2,random_state=1)
X_training=X_training.T
Y_training=Y_training.T
Xtest=Xtest.T
Ytest=Ytest.T

RowZ_parameters = train_or_load_model(
        X_training, 
        Y_training, 
        hidden_layers=(16, 16, 16),
        filename='recTp/RowZ_model.pkl',
        force_train=False, 
        learning_rate=1, 
        n_iter=10000
    )
ytraining_pred = predict(X_training, RowZ_parameters)
ytest_pred = predict(Xtest, RowZ_parameters)
print("Predictions on the training set after optimization:", ytraining_pred.flatten())
print("Actual:", Y_training.flatten())
print("Accuracy:", accuracy_score(Y_training.flatten(), ytraining_pred.flatten()))

print("Predictions on the test set after optimization :", ytest_pred.flatten())
print("Actual:", Ytest.flatten())
print("Accuracy:", accuracy_score(Ytest.flatten(), ytest_pred.flatten()))
