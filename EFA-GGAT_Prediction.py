
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import folium
from branca.colormap import linear
from pykrige.ok import OrdinaryKriging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

import random, os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas.core.groupby')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# Fixed random number
# ==============================
SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================
# Parameter
# ==============================
PROJECTED_CRS = "EPSG:32647"
GRID_SIZE, MIN_SAMPLES = 3000, 1
EPOCHS, LR, K_NEIGHBORS = 300, 1e-3, 8   

# ==============================
# Read data & process elements
# ==============================
df = pd.read_excel('Processed_geochemical_elements.xlsx')
df['label'] = df['Type'].apply(lambda x: 0 if str(x).lower()=='barren' else 1)
df_all = df.copy()

major_elements = [
    'clr_MnO','clr_CaO', 'clr_Na2O', 'clr_K2O','clr_P2O5','clr_TiO2','clr_MgO','clr_SiO2','clr_Al2O3','clr_TFe2O3'
]
trace_elements = [
    'log_Be','log_Cu', 'log_Ba', 'log_Y','log_Zr', 'log_Hf', 'log_Pb', 'log_U','log_Ga','log_V', 'log_Cr', 'log_Ni',
    'log_Sr','log_Sc','log_Co','log_Rb','log_Ta','log_Nb','log_Li','log_Th','log_Zn'
]
ree_elements = [
    'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
]
feature_cols = [c for c in (major_elements + trace_elements + ree_elements) if c in df.columns]

# Project all points
gdf_all = gpd.GeoDataFrame(
    df_all,
    geometry=[Point(xy) for xy in zip(df_all['Longitude'], df_all['Latitude'])],
    crs="EPSG:4326"
).to_crs(PROJECTED_CRS)

# ==============================
# Building a rules grid
# ==============================
xmin, ymin, xmax, ymax = gdf_all.total_bounds
polygons, grid_ids = [], []
for gid, (x, y) in enumerate([(x, y) for x in np.arange(xmin, xmax+GRID_SIZE, GRID_SIZE)
                                    for y in np.arange(ymin, ymax+GRID_SIZE, GRID_SIZE)]):
    polygons.append(Polygon([(x,y),(x+GRID_SIZE,y),(x+GRID_SIZE,y+GRID_SIZE),(x,y+GRID_SIZE)]))
    grid_ids.append(gid)
grid = gpd.GeoDataFrame({'grid_id': grid_ids}, geometry=polygons, crs=PROJECTED_CRS)

points_with_grid = gpd.sjoin(gdf_all, grid[['grid_id','geometry']], how='inner', predicate='within')

def filter_grid_points(group):
    if (group['label'] == 1).any():
        return group[group['label'] == 1]
    else:
        return group
points_filtered = points_with_grid.groupby('grid_id', group_keys=False).apply(filter_grid_points)

agg_label = points_filtered.groupby('grid_id')['label'].agg(total_count='count', pos_count='sum').reset_index()
agg_label['neg_count'] = agg_label['total_count'] - agg_label['pos_count']
agg_label['pos_ratio'] = agg_label['pos_count'] / agg_label['total_count']

agg_log = points_filtered.groupby('grid_id')[feature_cols].mean().reset_index()
grid_features = grid.merge(agg_label, on='grid_id', how='left').merge(agg_log, on='grid_id', how='left')
grid_features[feature_cols] = grid_features[feature_cols].fillna(0)
for c in ['total_count', 'pos_count', 'neg_count']:
    grid_features[c] = grid_features[c].fillna(0)
grid_features['has_samples'] = grid_features['total_count'] >= MIN_SAMPLES

# ==============================
# Kriging interpolation initialises a grid without samples
# ==============================
known, unknown = grid_features[grid_features['total_count'] > 0].copy(), grid_features[grid_features['total_count'] == 0].copy()
for df_ in [known, unknown]:
    df_['center_x'], df_['center_y'] = df_.geometry.centroid.x, df_.geometry.centroid.y

for col in feature_cols:
    OK = OrdinaryKriging(known['center_x'], known['center_y'], known[col].values,
                         variogram_model='spherical', verbose=False, enable_plotting=False)
    z_pred, _ = OK.execute('points', unknown['center_x'], unknown['center_y'])
    unknown[col] = z_pred

all_grids = pd.concat([known, unknown]).reset_index(drop=True)

# ==============================
# Feature standardisation (based on known grids)
# ==============================
scaler = StandardScaler()
X_known = known[feature_cols].values
scaler.fit(X_known)
X_known_scaled = scaler.transform(X_known)
X_unknown_scaled = scaler.transform(unknown[feature_cols].values)
X_all_scaled = np.vstack([X_known_scaled, X_unknown_scaled])
X_all = torch.tensor(X_all_scaled, dtype=torch.float32)

all_grids = pd.concat([known, unknown], ignore_index=True)

# ==============================
# Construct KNN graph
# ==============================
def build_knn_graph(centers, k=8):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(centers)
    idx = nbrs.kneighbors(centers)[1]
    edges = [[i,j] for i in range(len(centers)) for j in idx[i] if i!=j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_index = build_knn_graph(all_grids[['center_x','center_y']].values, k=K_NEIGHBORS)
train_mask = torch.zeros(X_all.shape[0], dtype=torch.bool)
train_mask[:len(known)] = True
y_known = torch.tensor(known['pos_ratio'].values, dtype=torch.float32)

# ==============================
# EFA-GGAT
# ==============================
class FeatureEmbedding(nn.Module):
    """Learn an embedding vector for each feature and integrate it with numerical values"""
    def __init__(self, num_features, d_model):
        super().__init__()
        self.feature_embed = nn.Embedding(num_features, d_model)
        self.value_proj = nn.Linear(1, d_model, bias=False)

    def forward(self, x):  # x: (N, F)
        val = x.unsqueeze(-1)                     # (N, F, 1)
        idx = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        emb = self.feature_embed(idx)             # (N, F, d_model)
        val_proj = self.value_proj(val)           # (N, F, d_model)
        return emb * val_proj

class FeatureAttention(nn.Module):
    """Multi-head self-attention on feature dimensions, learning the combinations between elements"""
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):   # x: (N, F, d_model)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class EFA_GGAT_Regression(nn.Module):
    """Output the value of a single node (0~1)"""
    def __init__(self, num_features, d_model=32, hidden=32, heads=4, dropout=0.6):
        super().__init__()
        self.embedding = FeatureEmbedding(num_features, d_model)
        self.feature_attn = FeatureAttention(d_model, nhead=4, dropout=dropout)
        self.proj = nn.Linear(d_model, hidden * heads)


        self.gat1 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.gate1 = nn.Sequential(nn.Linear(hidden * heads, hidden * heads), nn.Sigmoid())
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        self.gate2 = nn.Sequential(nn.Linear(hidden * heads, hidden * heads), nn.Sigmoid())
        self.gat_out = GATConv(hidden * heads, 1, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        emb = self.embedding(x)                  # (N, F, d_model)
        feat_attn = self.feature_attn(emb)       # (N, F, d_model)
        h = feat_attn.mean(dim=1)                # (N, d_model)
        h = self.proj(h)                         # (N, hidden*heads)

        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = self.gate1(h) * h
        h = F.dropout(h, p=0.6, training=self.training)

        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = self.gate2(h) * h
        h = F.dropout(h, p=0.6, training=self.training)

        out = self.gat_out(h, edge_index)        # (N, 1)
        return torch.sigmoid(out).squeeze()      # (N,)

# ==============================
# Train the model
# ==============================
model = EFA_GGAT_Regression(num_features=X_all.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    pred = model(X_all, edge_index)
    loss = loss_fn(pred[train_mask], y_known)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

# Prediction
model.eval()
with torch.no_grad():
    all_grids['pred_ratio'] = model(X_all, edge_index).numpy()
all_grids['display_ratio'] = all_grids['pos_ratio']
all_grids.loc[all_grids['total_count']==0, 'display_ratio'] = all_grids.loc[all_grids['total_count']==0, 'pred_ratio']

# ==============================
# Save the latitude and longitude of each cell and the predicted values to Excel
# ==============================
output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

all_grids_wgs = all_grids.to_crs(epsg=4326)

grid_export = all_grids_wgs.copy()

if grid_export.crs is not None and grid_export.crs.is_geographic:
    grid_proj = grid_export.to_crs(grid_export.estimate_utm_crs())
    centroids = grid_proj.geometry.centroid.to_crs(grid_export.crs)
else:
    centroids = grid_export.geometry.centroid

grid_export['center_lon'] = centroids.x
grid_export['center_lat'] = centroids.y


export_cols = [
    'grid_id',           # Grid Number
    'center_lon',        # Central longitude
    'center_lat',        # Central latitude
    'pred_ratio',        # Mineralisation potential (0~1)
]
export_df = grid_export[export_cols].copy()

excel_path = os.path.join(output_dir, "EFA-GGAT Prediction.xlsx")
export_df.to_excel(excel_path, index=False)
print(f"The grid prediction results have been saved to: {excel_path}")