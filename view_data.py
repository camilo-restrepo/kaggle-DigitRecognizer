import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


dataset = pd.read_csv("data/test.csv")
train = dataset.iloc[:, :].values
train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)

plt.imshow(train[1][0], cmap=cm.binary, interpolation='none')
plt.show()
