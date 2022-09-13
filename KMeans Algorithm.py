#   It is a type of unsupervised machine learning, which
# means that the alogorithm only trains on inputs and no output.

#   It finds the distinct groups of data that are closest together.
# Specifically, it partitions the data into clusters in which each points falls 
# into a cluster whose mean is closest to that data point.  

from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


#   As an example we will use age and spending score as the inputs for our KMeans algorithm 

df = pd.read_csv(r"c:\Users\Minato132\Desktop\Code\Modern Code\Data\Mall_Customers.csv")

x = df[['Age', 'Spending Score (1-100)']].copy()

#   Now we need to determine the amount of clusters we will use. We can use the elbow method
# Elbow method: (within-cluster-sum-of=squares)
# We first need to initialize an empty list to append the values to 

#wcss = []

#   We need to define a for loop that containes instances of the K-means class.


#for i in range(1, 11):
 #   kmeans = KMeans(n_clusters  = i, random_state = 0)
  #  kmeans.fit(x)
   # wcss.append(kmeans.inertia_)

#   Here a couple of things need to be noted in this line of code. 
# 1) random_state : This is like the .seed(0) method in random.seed(0)
# 2) n_clusters: This is to signify how many clusters do we want to define.  
# 3) .fit() is a method in scikit-learn that will basically perform a
#    linear regression on the data points
# 4) The .inerta_ is to tell the program to add in the inertia values of each of your clusters
#    This will be important as we need this information to tell which size cluster are optimal 

#sns.set()
#plt.plot(range(1,11), wcss)

#plt.title('Selecting the number of clusters using the elbow method')
#plt.xlabel('Clusters')
#plt.ylabel('WCSS')
#plt.show()

#   The graph that is showing is exactly the elbow method. 
# Basially, we are trying to choose an optimal value for the amount of clusters
# to represent our data. The way to find our optimal value is to look at the graph and
# find the point where the inertia decreases in a linear way. Here it is k = 4


#___________________________________________________________________________________________


#   Now we will try to visualize these clusters and their location, in our data. 
# We will rerun the KMeans method but with known cluster amount this time. The optimal cluster
# that should have been found in the last section.

kmeans = KMeans(n_clusters=4)

#   Next I am going to have the fit_predict() method in KMeans give me the labels of
# which data point belongs to which cluster 

label = kmeans.fit_predict(x)

#   The fit_predict will filter each value into labels, each of which is placed in an array. 
#  Each of these labels are tied to values in the original dataset. All we have to do now 
#  is access each of those datasets by calling on which dataset has a certain label  

label0 = x[label == 0]
label1 = x[label == 1]
label2 = x[label == 2]
label3 = x[label == 3]

#   Now all we do is plot each of these labels

sns.set()
plt.figure()
plt.scatter(label0['Age'], label0['Spending Score (1-100)'], color = 'red')
plt.scatter(label1['Age'], label1['Spending Score (1-100)'], color = 'blue')
plt.scatter(label2['Age'], label2['Spending Score (1-100)'], color = 'black')
plt.scatter(label3['Age'], label3['Spending Score (1-100)'], color = 'green')
plt.xlabel('Spending Score')
plt.ylabel('Age')
plt.show()

# This type of info are useful to retail companies looking to target specific consumer
# demographics. 