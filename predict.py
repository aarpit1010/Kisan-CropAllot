import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
data = np.array(['Kharif     ','Jowar',4000])
data = pd.DataFrame(data.reshape(-1,3), columns=['Season','Crop','Area'])
le1 = LabelEncoder()
le1.classes_ = np.load('data/season.npy', allow_pickle=True)
data['Season'] = le1.transform(data['Season'])
print({index: label for index , label in enumerate(le1.classes_)})
le2 = LabelEncoder()
le2.classes_ = np.load('data/crop.npy', allow_pickle=True)
data['Crop'] = le2.transform(data['Crop'])
file = open('models/random_forest_1.sav', 'rb')
model = pkl.load(file)
pred = model.predict(data)
print(pred)
file.close()
