import pandas as pd
import pickle

# load trained model
file_to_open = open("data/models/regression_lr.pickle", 'rb')
trained_model = pickle.load(file_to_open)
file_to_open.close()

# load data that we want predictions for
# #es muss noch ein 'prediction-data.csv' file geben hierfür; vom train.py generiert, aber wo?
prediction_data = pd.read_csv('data/prediction-data.csv', sep=";")

print(trained_model.predict(prediction_data))
