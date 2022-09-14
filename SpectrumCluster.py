#   This type of clustering method is good for high 
# dimensional and complex data. 

from sklearn.cluster import SpectralClustering
import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv(r"c:\Users\Minato132\Desktop\Code\Modern Code\Data\Mall_Customers.csv")

x = df[['Age', 'Spending Score (1-100)']].copy()

sc = SpectralClustering(n_clusters= 5, random_state=0, n_neighbors=8, affinity='nearest_neighbors')

sc.fit(x) 

labels = sc.fit_predict(x)

for i in range(0, 4): 
    data = x[labels == i]
    plt.scatter(data['Age'], data['Spending Score (1-100)'])

plt.show()
    