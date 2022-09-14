#    We explore the Gaussian Mixture Model. Like the name suggests
# we only only apply this to data sets that have a Gaussian 
# Distribution. 

#   They are good in describing things such as popluation heights
# and weights. 

#   These distributions are inherently useful because they have
# well-defined properties such as mean, variance and co-variance.

#   GMM is a more ideal method for data sets of moderate size and
# and complexity because it can capture complex clusters, rather 
# than the sphereical clusters

from sklearn.mixture import GaussianMixture 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv(r"c:\Users\Minato132\Desktop\Code\Modern Code\Data\Mall_Customers.csv")

x = df[['Age', 'Spending Score (1-100)']].copy()

cluster = 5
GMM = GaussianMixture(n_components = cluster, random_state = 0 )
GMM.fit(x) 

labels = GMM.predict(x)

for i in range(0, cluster):
    data = x[labels == i]
    plt.scatter(data['Age'], data['Spending Score (1-100)'])

plt.show() 
