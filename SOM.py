import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

#Importing the dataset
dataset = pd.read_csv(r'Fantasy Football Dataset - Sheet1.csv')
X = dataset.iloc[:, :].values
y = dataset.iloc[:, -1].values
player_name = dataset.iloc[:, 1].values

#Label Encoding
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])


#Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#Training the SOM
som = MiniSom(x=10, y=10, input_len=18, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
        w[1] + 0.5,
        markers[y[i]],
        markeredgecolor=colors[y[i]],
        markerfacecolor='None',
        markersize=10,
        markeredgewidth=2)
show()

#Finding the player
mapping = som.win_map(X)
#Change the coordinates in the mapping dictionary '1, 1' and '4, 1' to the outlined winning nodes (show as a white square on the graph) 
player = np.concatenate((mapping[(1, 1)], mapping[(4, 1)]), axis=0).reshape(-18, 18)
player = sc.inverse_transform(player)

#Printing the Fraund Clients
print('Players To Pick')
for i in player[:, 1]:
    #print(int(i))
    print(player_name[int(i)])