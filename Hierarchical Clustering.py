# Importing the necessary libraries
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation
import warnings  # To handle warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Importing the dataset
dataset = pd.read_csv(r"D:\FSDS Material\Dataset\Clustering\Mall_Customers.csv")  # Load the dataset
# Selecting columns 'Annual Income' and 'Spending Score' for clustering
x = dataset.iloc[:, [3, 4]].values  

# Creating a dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

# Generate the linkage matrix using Ward's method and plot the dendrogram
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))  
plt.title('Dendrogram')  # Add a title to the dendrogram
plt.xlabel('Customers')  # Label for the x-axis representing customers
plt.ylabel('Euclidean distances')  # Label for the y-axis representing distances
plt.show()  # Display the dendrogram plot

# Fitting Agglomerative Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering

# Create an instance of AgglomerativeClustering with 5 clusters
hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')  
y_hc = hc.fit_predict(x)  # Fit the model and predict cluster labels for each data point

# Visualizing the clusters
# Plotting each cluster with a different color
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='red', label='Cluster 1')  # Cluster 1
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')  # Cluster 2
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='green', label='Cluster 3')  # Cluster 3
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')  # Cluster 4
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')  # Cluster 5
plt.title('Clusters of customers')  # Add a title
plt.xlabel('Annual Income (k$)')  # Label for the x-axis
plt.ylabel('Spending Score (1-100)')  # Label for the y-axis
plt.legend()  # Add a legend to identify clusters
plt.show()  # Display the plot
