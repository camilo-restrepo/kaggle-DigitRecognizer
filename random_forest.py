from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

training_dataset = pd.read_csv("data/train.csv")

labels = training_dataset[[0]].values.ravel()
train = training_dataset.iloc[:, 1:].values

rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf.fit(train, labels)

test = pd.read_csv("data/test.csv").values
pred = rf.predict(test)

np.savetxt('submission/random_forest.csv', np.c_[range(1, len(test)+1), pred], delimiter=',', header='ImageId,Label',
           comments='', fmt='%d')
