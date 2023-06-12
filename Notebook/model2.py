# **Import Library**
"""

import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

#Load Dataset
filename_df = pd.read_csv("/content/Data Rumah Sakit di Surakarta - Sheet2.csv")

#Menampilkan 5 data teratas
filename_df.head()

#Melihat informasi data
filename_df.info()

#Mengganti tipe data 'Kode Rumah Sakit'
filename_df['Kode Rumah Sakit'] = filename_df['Kode Rumah Sakit'].astype(str)

#Melihat informasi data
filename_df.info()

"""# **Data Cleansing**"""

#Mengecek missing value pada data 
miss_value = filename_df.isna().mean(axis=0)
miss_value

#Menghapus kolom yang mengandung missing value
filename_drop = filename_df.drop(['Email', 'Kamar'], axis = 1)
filename_drop.head()

"""# **Data Preparation**"""

#Mengecek outlier
data_num = ['Longitude', 'Latitude']
sns.boxplot(filename_drop[data_num])

#Melakukan standarisasi dan scaling pada Longitude dan Latitude
filename_num = filename_drop[data_num]
data_scaled = pd.DataFrame(StandardScaler().fit_transform(filename_num))
data_scaled.columns = ['Longitude', 'Latitude']
data_scaled

#Menghapus kolom yang mengandung Longitude dan Latitude
filename = filename_drop.drop(['Longitude', 'Latitude'], axis = 1)
filename

#Menggabungkan data hasil scaling kedalam filename
filename[data_scaled.columns] = data_scaled[data_scaled.columns]
filename.head()

#Mengurutkan data berdasarkan Longitude, Latitude, Tipe, dan BPJS
data_hosp = filename.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Build Model**

**Mencari nilai K optimal**
"""

#Model dengan inisiasi K=3
X = KMeans(n_clusters = 3)
X.fit(data_scaled)

#Label cluster
y = X.labels_
y

#Analisis dengan menggunakan metode Elbow

distortions = []
K = range(1,10)
for k in K:
  X = KMeans(n_clusters = k)
  X.fit(data_scaled)
  inertia_score = X.inertia_
  distortions.append(inertia_score)

#Plot Elbow untuk setiap cluster
plt.plot(K, distortions, marker = 'o')
plt.xlabel('k')
plt.ylabel('distortions')
plt.title('The Optimal k')
plt.show()

#Analisis dengan menggunakan metode Silhouette
silhouette = []

K = range(2, 10)
for k in K:
  X = KMeans(n_clusters = k, init = 'k-means++')
  X.fit(data_scaled)
  labels = X.labels_
  silhouette.append(metrics.silhouette_score(data_scaled, labels, metric = 'euclidean'))

#Melihat silhouette score dari setiap cluster
sil_score = pd.DataFrame({'Cluster' : K, 'Score' : silhouette})
sil_score

#Model final dengan K-optimal=5
coordinates = filename_drop[['Longitude', 'Latitude']]

y = KMeans(n_clusters = 5, init = 'k-means++')
labels = y.fit(coordinates).labels_
print('k = 5', 'silhouette_score', metrics.silhouette_score(coordinates, labels, metric = 'euclidean'))

#Label cluster
y.labels_

#Memberi label cluster untuk setiap data
filename_drop['Cluster'] = y.predict(filename_drop[['Longitude', 'Latitude']])
filename_drop.head()

#Mengurutkan data yang telah diberi label berdasarkan Longitude, Latitude, Tipe, dan BPJS
data_hosp = filename_drop.sort_values(by=['Longitude', 'Latitude', 'Tipe', 'BPJS'], ascending = False)
data_hosp.head()

"""# **Test Model**"""

def recommended_hospitals(filename_drop, Longitude, Latitude):
  #Prediksi cluster
  Cluster = y.predict(tf.reshape(np.array([Longitude, Latitude]), (1, -1)))[0]
  data_cluster = filename_drop[filename_drop['Cluster']==Cluster].copy()
  print(Cluster)

  #Menghitung jarak user dengan rumah sakit
  data_cluster['Hospital Distance'] = data_cluster.apply(lambda x: euclidean_distances([[x.Longitude, x.Latitude]], [[Longitude, Latitude]])*1000, axis = 1)

  #Mengurutkan hasil rekomendasi
  col = ['Longitude', 'Latitude', 'Tipe', 'BPJS']
  data_cluster.sort_values(col, ascending = [False]*len(col), inplace = True)

  #Pilih jumlah N rekomendasi
  n = 50
  data_cluster = data_cluster.iloc[:n]
  return data_cluster

Longitude = 110.85664241141414
Latitude = -7.5523142693779475
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp

Longitude = 110.82716015648799
Latitude = -7.554622333033178
recommend_hosp = recommended_hospitals(data_hosp, Longitude, Latitude)
recommend_hosp