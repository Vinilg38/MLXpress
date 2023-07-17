from MLXpress.iris import clustering,vis_clusters
from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
clustering(model,2)
vis_clusters(model)