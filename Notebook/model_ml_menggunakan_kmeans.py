import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

filename_df = pd.read_csv("/content/Data Rumah Sakit di Surakarta - Sheet2.csv")
filename_df.head()

filename_df.info()

filename_df.shape

filename_drop = filename_df.drop_duplicates(['Nama Rumah Sakit'], keep = 'first')
filename_drop

best_rs = filename_drop.sort_values(by=['Tipe', 'BPJS'], ascending = False)
best_rs

from sklearn.cluster import KMeans
from sklearn import metrics

# Mencari nilai K
coordinates = filename_drop[['Longitude', 'Latitude']]

distortions = []
K = range(1,10)
for k in K:
  kmeansModel = KMeans(n_clusters = k)
  kmeansModel = kmeansModel.fit(coordinates)
  distortions.append(kmeansModel.inertia_)

score_dist = pd.DataFrame({'Cluster' : K, 'Value' : distortions})
score_dist

plt.plot(K, distortions, marker = 'o')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('the elbow method showing the optimal k')
plt.show()

silhouette = []

K = range(2, 11)
for k in K:
  labels = KMeans(n_clusters = k, init = 'k-means++').fit(coordinates).labels_
  silhouette.append(metrics.silhouette_score(coordinates, labels, metric = 'euclidean'))

sil_score = pd.DataFrame({'Cluster' : K, 'Score' : silhouette})
sil_score

from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize = (15, 8))
K = [2, 3, 4, 5]
for k in K:
  kmeans = KMeans(n_clusters = k, init = 'k-means++')
  q, mod =divmod(k, 2)
  visualizer = SilhouetteVisualizer(kmeans, colors = 'yellowbrick', ax = ax[q-1][mod])
  visualizer.fit(coordinates)

labels = KMeans(n_clusters = 2, init = 'k-means++').fit(coordinates).labels_
print('k = 2', 'silhouette_score', metrics.silhouette_score(coordinates, labels, metric = 'euclidean'))

filename_drop['cluster'] = kmeans.predict(filename_drop[['Longitude', 'Latitude']])
filename_drop

best_rs = filename_drop.sort_values(by=['Tipe', 'BPJS'], ascending = False)
best_rs

def recommended_hospitals(filename_drop, Longitude, Latitude):
  cluster = kmeans.predict(np.array([Longitude, Latitude]). reshape(1, -1))[0]
  print(cluster)
  return filename_drop[filename_drop['cluster']==cluster].iloc[:10][['Nama Rumah Sakit', 'Latitude', 'Longitude']]

rec_hosp = recommended_hospitals(best_rs, 110.85664241141414, -7.5523142693779475)
rec_hosp

rec_hosp = recommended_hospitals(best_rs, 110.82712799471938, -7.554579806884118)
rec_hosp

