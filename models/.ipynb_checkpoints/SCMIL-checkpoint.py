import torch
from torch import nn, einsum
from cuml import KMeans
import cupy as cp
import torch.nn.functional as F
from einops import rearrange, reduce
import math
from .SoftFilter import SoftFilterLayer
from .AMIL import AMIL_layer
import numpy as np

class HierarchicalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_prob=0.25):
        super(HierarchicalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout_prob)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        if x.shape[1] == 0:
            raise ValueError("Input sequence length is zero")
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        x = x.transpose(0, 1)
        return x

class HierarchicalClusterLocalAttention(nn.Module):
    def __init__(self, hidden_size=384, n_cluster=None, cluster_size=None, feature_weight=0, dropout_rate=0.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        
        # Learnable weights for morphological and spatial features
        self.weight_params = nn.Parameter(torch.tensor([0.8, 0.2]), requires_grad=True)  # Initialize w1=0.8, w2=0.2
        
        self.local_attention = HierarchicalSelfAttention(dim=hidden_size, num_heads=8, dropout_prob=dropout_rate)
        self.global_attention = HierarchicalSelfAttention(dim=hidden_size, num_heads=8, dropout_prob=dropout_rate)

    def cluster_patches(self, x, coords):
        B, L, C = x.shape
        # print(f"Cluster patches: x shape={x.shape}, device={x.device}, coords shape={coords.shape}, device={coords.device}")
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaN or Inf values")
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            raise ValueError("Input coords contains NaN or Inf values")
        
        if coords.shape != (B, L, 2):
            raise ValueError(f"Expected coords shape {(B, L, 2)}, got {coords.shape}")
        # print('n_cluster',self.n_cluster)
        # print('cluster_size',self.cluster_size)
        if self.n_cluster is None and self.cluster_size is not None:
            n_cluster = max(1, L // self.cluster_size)
        elif self.n_cluster is not None:
            n_cluster = self.n_cluster
        else:
            raise NotImplementedError

        if L < n_cluster:
            n_cluster = max(1, L)
            # print(f"Adjusted n_cluster to {n_cluster} as L={L}")

        if L == 0:
            raise ValueError("No patches to cluster (L=0)")

        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(self.weight_params, dim=0)
        w_morph = weights[0]  # Morphological weight (w1)
        w_spatial = weights[1]  # Spatial weight (w2)
        # print(f"Learned weights: w_morph={w_morph.item():.4f}, w_spatial={w_spatial.item():.4f}")

        if self.feature_weight == -1:
            labels = torch.randint(0, n_cluster, (L,), device=x.device, dtype=torch.long).cpu().numpy()
            # print(f"Random labels assigned: {labels[:10]}...")
        else:
            with torch.no_grad():
                # Normalize features and coordinates on GPU
                feats = x.squeeze(0)  # Shape: [L, C]
                coords_norm = coords.squeeze(0)  # Shape: [L, 2]
                feats_norm = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-6)
                coords_norm = (coords_norm - coords_norm.mean(dim=0)) / (coords_norm.std(dim=0) + 1e-6)
                feats_norm = F.normalize(feats_norm, p=2, dim=1)  # L2 normalize
                # print(f"feats_norm shape={feats_norm.shape}, device={feats_norm.device}")
                # print(f"coords_norm shape={coords_norm.shape}, device={coords_norm.device}")
                
                # Batch processing for similarity computation
                batch_size = 1000  # Adjust based on GPU memory
                k = min(10, L - 1)  # Number of neighbors
                indices = torch.zeros(L, k, device=x.device, dtype=torch.long)
                values = torch.zeros(L, k, device=x.device, dtype=torch.float32)
                
                for i in range(0, L, batch_size):
                    end = min(i + batch_size, L)
                    feats_batch = feats_norm[i:end]  # Shape: [batch_size, C]
                    coords_batch = coords_norm[i:end]  # Shape: [batch_size, 2]
                    
                    # Cosine similarity
                    cosine_sim = torch.matmul(feats_batch, feats_norm.T)  # Shape: [batch_size, L]
                    
                    # Euclidean distance
                    dist = torch.cdist(coords_batch, coords_norm, p=2)  # Shape: [batch_size, L]
                    sigma = dist.std() + 1e-6  # Scaling factor
                    spatial_sim = torch.exp(-dist / sigma)  # Shape: [batch_size, L]
                    
                    # Combine similarities
                    similarity = w_morph * cosine_sim + w_spatial * spatial_sim
                    del cosine_sim, dist, spatial_sim  # Free memory
                    
                    # Get top-k neighbors
                    topk_vals, topk_idx = torch.topk(similarity, k, dim=1)  # Shapes: [batch_size, k]
                    indices[i:end] = topk_idx
                    values[i:end] = topk_vals
                    del similarity, topk_vals, topk_idx  # Free memory
                
                # Build sparse k-NN graph
                knn_graph = torch.zeros(L, k, device=x.device, dtype=torch.float32)
                knn_graph = values / (values.sum(dim=1, keepdim=True) + 1e-6)  # Normalize
                # print(f"knn_graph shape={knn_graph.shape}, device={knn_graph.device}")
                
                # Convert to cupy for cuml.KMeans
                knn_graph_cupy = cp.asarray(knn_graph)
                # print(f"knn_graph_cupy shape={knn_graph_cupy.shape}, dtype={knn_graph_cupy.dtype}")
                
                # Use K-Means on graph-based representation
                try:
                    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', tol=1e-4, max_iter=50, random_state=1)
                    labels = kmeans.fit_predict(knn_graph_cupy)
                    labels = cp.asnumpy(labels)  # Convert to numpy for indexing
                    # print(f"KMeans labels: {labels[:10]}...")
                    # print('length of labels',len(labels))
                except Exception as e:
                    print(f"Graph-based clustering failed: {e}, falling back to random clustering")
                    labels = np.random.randint(0, n_cluster, size=L)
                    print(f"Random labels: {labels[:10]}...")
                
                # Clear GPU memory
                torch.cuda.empty_cache()
        
        if not (labels >= 0).all() or not (labels < n_cluster).all():
            raise ValueError(f"Invalid labels: min={labels.min()}, max={labels.max()}, n_cluster={n_cluster}")
        
        index = np.argsort(labels, kind='stable').tolist()
        # print(f"Index: {index[:10]}..., length={len(index)}")
        if max(index, default=-1) >= L or min(index, default=L) < 0:
            raise ValueError(f"Invalid index values: min={min(index)}, max={max(index)}, L={L}")
        
        window_sizes = np.bincount(labels).tolist()
        window_sizes_new = []
        for size in window_sizes:
            # print('size is',size)
            if size >= (self.cluster_size * 2 if self.cluster_size else 2):
                num_splits = max(1, size // (self.cluster_size or 1))
                quotient = size // num_splits
                remainder = size % num_splits
                result = [quotient + 1 if i < remainder else quotient for i in range(num_splits)]
                window_sizes_new.extend(result)
            else:
                window_sizes_new.append(size)
        window_sizes = window_sizes_new
        # print(f"Window sizes: {window_sizes}, sum={sum(window_sizes)}")
        
        clusters = []
        coord_clusters = []
        index_tensor = torch.tensor(index, device=x.device, dtype=torch.long)
        x = x[:, index_tensor]
        coords = coords[:, index_tensor]
        # print(f"After indexing: x shape={x.shape}, device={x.device}, coords shape={coords.shape}, device={coords.device}")
        
        now = 0
        for size in window_sizes:
            if size > 0:
                # print('size is is',size)
                clusters.append(x[:, now:now + size])
                coord_clusters.append(coords[:, now:now + size])
                now += size
        
        if not clusters:
            raise ValueError("No valid clusters formed")
        if sum(window_sizes) != L:
            raise ValueError(f"Sum of window sizes {sum(window_sizes)} does not match L={L}")
        
        return clusters, coord_clusters, labels

    def forward(self, x, coords=None, cluster_label=None, return_patch_label=False):
        B, L, C = x.shape
        # print('cluster_label:',cluster_label)
        # print('return_patch_label:',return_patch_label)
        # print('shape of x',x.shape)
        # print('shape of coords',coords.shape)
        if cluster_label is not None:
            labels = cluster_label.numpy()[0]
            clusters, _, _ = self.cluster_patches(x, coords)
        else:
            clusters, _, labels = self.cluster_patches(x, coords)
        
        refined_clusters = []
        for L_i in clusters:
            L_i_prime = self.local_attention(L_i)
            refined_clusters.append(L_i_prime)
        
        cluster_reps = []
        for L_i_prime in refined_clusters:
            R_i = torch.mean(L_i_prime, dim=1)
            cluster_reps.append(R_i)
        R = torch.stack(cluster_reps, dim=1)
        # print(f"R shape: {R.shape}")
        
        R_prime = self.global_attention(R)
        # print(f"R_prime shape: {R_prime.shape}")
        
        h = torch.cat(refined_clusters, dim=1)
        # print(f"h shape: {h.shape}")
        R_prime_expanded = R_prime.mean(dim=1, keepdim=True).expand(-1, h.shape[1], -1)
        # print(f"R_prime_expanded shape: {R_prime_expanded.shape}")
        h = h + R_prime_expanded
        # print(f"h shape: {h.shape}")
        if return_patch_label:
            return h, labels
        return h, None

class SCMIL(nn.Module):
    def __init__(self, n_classes=4, input_size=384, hidden_size=None, deep=1, n_cluster=None, cluster_size=None,
                 feature_weight=0, as_backbone=False, dropout_rate=0.25, with_softfilter=False, use_filter_branch=False,
                 with_cssa=True, softfilter_quantile=0.5, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.deep = deep
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        self.as_backbone = as_backbone
        self.with_softfilter = with_softfilter
        self.with_cssa = with_cssa
        self.softfilter_quantile = softfilter_quantile
        if self.with_softfilter:
            self.use_filter_branch = True
        else:
            self.use_filter_branch = use_filter_branch
        
        if self.with_softfilter:
            self.softfilter = SoftFilterLayer(dim=input_size, hidden_size=256, deep=1, quantile=softfilter_quantile)
        
        if hidden_size is None:
            hidden_size = input_size
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        if with_cssa:
            layers = []
            for i in range(deep):
                layers.append(HierarchicalClusterLocalAttention(
                    hidden_size=hidden_size,
                    n_cluster=n_cluster,
                    cluster_size=cluster_size,
                    feature_weight=feature_weight,
                    dropout_rate=dropout_rate
                ))
            self.attens = nn.Sequential(*layers)
        self.amil = AMIL_layer(hidden_size, 256, dropout=dropout_rate)
        if not self.as_backbone:
            self.classifier = nn.Linear(hidden_size, n_classes)
        
        self.iter = 0

    def forward(self, x, coords=None, cluster_label=None, return_IS=False, return_patch_label=False):
        self.iter += 1
        res = {}
        if self.with_softfilter:
            x, logits, idx_high, idx_low, threshold = self.softfilter(x)
            # print(f"Logits shape: {logits.shape}, device={logits.device}, threshold={threshold.item()}")
            if return_IS:
                res['IS'] = logits.squeeze().cpu().detach().numpy()
                res['threshold'] = threshold.item()
        
        if self.use_filter_branch:
            h = x[:, idx_high]
            coords_filtered = coords[:, idx_high] if coords is not None else None
            h_app = x[:, idx_low]
            # print('lol',h.shape)
            # print('lolwat',h_app.shape)
        else:
            h = self.fc1(x)
            coords_filtered = coords
            # print(f"Projected h shape: {h.shape}")
        
        if self.with_cssa:
            if h.shape[1] > 1:
                for atten in self.attens:
                    h, patch_label = atten(h, coords_filtered, cluster_label, return_patch_label)
            else:
                patch_label = None
                # print(f"Single patch, skipping attention: h shape={h.shape}")
        
        if return_patch_label:
            patch_label_all = np.zeros(len(logits.squeeze()) if self.use_filter_branch else h.shape[1]) - 1
            if h.shape[1] > 1:
                if self.use_filter_branch:
                    patch_label_all[idx_high.cpu().numpy()] = patch_label
                else:
                    patch_label_all = patch_label
            res['patch_label'] = patch_label_all
            # print(f"Patch labels shape: {patch_label_all.shape}")
            return res
        
        if return_IS:
            # print(f"Returning IS: {res['IS'].shape}, threshold={res['threshold']}")
            return res
        
        if self.use_filter_branch:
            h = torch.cat([h, h_app], dim=1)
            # print(f"Concatenated h shape: {h.shape}")
        
        h, att, _ = self.amil(h)
        # print(f"AMIL output: h shape={h.shape}, att shape={att.shape}")
        if self.as_backbone:
            return h
        
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        res = {
            'logits': logits,
            'y_hat': Y_hat,
            'y_prob': Y_prob,
        }
        # print(f"Output: logits shape={logits.shape}, Y_hat shape={Y_hat.shape}, Y_prob shape={Y_prob.shape}")
        return res