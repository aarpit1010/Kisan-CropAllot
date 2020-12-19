import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error


def train_model(dist_name):
    """
    Arguments : Name of the District for which crop production is predicted
    LabelEncoder : Convert string to numeric with indexing starting from 0
    Model Prediction : Random Forest
    Model Save : pickle
    """
    data = pd.read_csv('data/apy.csv')
    
    data = data[data['District_Name']==dist_name]
    data = data.drop(columns = ['State_Name','District_Name','Crop_Year'])
    
    le1 = LabelEncoder()
    data['Season'] = le1.fit_transform(data['Season'])
    np.save('data/season.npy', le1.classes_)
    
    le2 = LabelEncoder()
    data['Crop'] = le2.fit_transform(data['Crop'])
    np.save('data/crop.npy', le2.classes_)
    
    data = data.dropna()
    target = data['Production']
    data = data.drop(columns = ['Production'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.20)
    
    model = RandomForestRegressor()
    
    model.fit(X_train, Y_train)
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    file = open('models/random_forest_1.sav', 'wb')
    pkl.dump(model, file)
    file.close()


