import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from minisom import MiniSom
import numpy as np
import plotly.express as px

data = pd.read_excel('StartUpsESG_0602.xlsx')

relevant_columns = ['Raised', 'ESG', 'E', 'S', 'G', 'Country', 'Year']

data = data[data['Year'] != 0]

data.dropna(inplace=True)

scaled_data = StandardScaler().fit_transform(data[['Raised', 'ESG', 'E', 'S', 'G']])
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

grid_size = (10, 10)  
sigma = 1.0
learning_rate = 0.5

som = MiniSom(grid_size[0], grid_size[1], pca_data.shape[1], sigma=sigma, learning_rate=learning_rate)
som.train_random(pca_data, 1000)

clusters = som.labels_map(pca_data, np.arange(len(data)))

cluster_data = pd.DataFrame([[k[0], k[1], v] for k, v in clusters.items()], columns=['x', 'y', 'Cluster'])
cluster_data['Cluster'] = cluster_data['Cluster'].astype(str)

data['Info'] = data.apply(lambda row: f'País: {row["Country"]}, Raised: {row["Raised"]}, ESG: {row["ESG"]}, Ano: {row["Year"]}', axis=1)

data = pd.concat([data, cluster_data], axis=1)

cluster_data.to_excel('clusters.xlsx', index=False)

fig = px.scatter(data, x='x', y='y', color='Country', hover_name='Info',
                 title='Clusters de Startups por Montante Levantado e ESG (MiniSom)',
                 labels={'x': 'Montante Levantado', 'y': 'Valor ESG', 'Cluster': 'Cluster'},
                 template='plotly_white')

fig.update_traces(selectedpoints=None)

fig.update_layout(
    clickmode='event+select',
    title='Selecione um país para análise'
)

fig.show()