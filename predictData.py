import numpy as np
from ric import predict, load_parameters  

model_file = 'recTP/original_model.pkl'
parameters = load_parameters(model_file)

new_data = np.loadtxt("recTp/new_data.txt")
X_new = new_data.T

y_pred = predict(X_new, parameters)
y_pred = y_pred.flatten()

data_with_labels = np.column_stack((new_data, y_pred))
np.savetxt("data_with_predictions.txt", data_with_labels,fmt="%.16f %.16f %.16f %d")