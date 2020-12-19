import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder

def allot_crops(dist_name, season, area):
    data = np.array([[season, area]])
    df = pd.read_csv('data/apy.csv')
    crops = df[df['District_Name']==dist_name]['Crop'].unique()
    data = pd.DataFrame(np.repeat(data, [len(crops)], axis=0), columns=['Season','Area'])
    data.insert(1, "Crop", crops)
    le1 = LabelEncoder()
    le1.classes_ = np.load('data/season.npy', allow_pickle=True)
    data['Season'] = le1.transform(data['Season'])
    le2 = LabelEncoder()
    le2.classes_ = np.load('data/crop.npy', allow_pickle=True)
    data['Crop'] = le2.transform(data['Crop'])
    file = open('models/random_forest_1.sav', 'rb')
    model = pkl.load(file)
    pred = model.predict(data)
    indices = np.argsort(pred)[::-1][:5]
    crops_alloted = le2.inverse_transform(data.iloc[indices]['Crop'])
    file.close()
    return crops_alloted

