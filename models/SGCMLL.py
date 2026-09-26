#######################################################for amil in egmll 2##########################################################

import torch
from torch import nn, einsum
from cuml import KMeans
import cupy as cp
import torch.nn.functional as F
from einops import rearrange, reduce
import math
from .DynamicFilter import DynamicFilterQuant
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
        self.weight_params = nn.Parameter(torch.tensor([0.8, 0.2]), requires_grad=True)  # w1=0.8, w2=0.2

        self.local_attention = HierarchicalSelfAttention(dim=hidden_size, num_heads=8, dropout_prob=dropout_rate)
        self.global_attention = HierarchicalSelfAttention(dim=hidden_size, num_heads=8, dropout_prob=dropout_rate)

    def cluster_patches(self, x, coords):
        B, L, C = x.shape
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input x contains NaN or Inf values")
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            raise ValueError("Input coords contains NaN or Inf values")

        if coords.shape != (B, L, 2):
            raise ValueError(f"Expected coords shape {(B, L, 2)}, got {coords.shape}")
        if self.n_cluster is None and self.cluster_size is not None:
            n_cluster = max(1, L // self.cluster_size)
        elif self.n_cluster is not None:
            n_cluster = self.n_cluster
        else:
            raise NotImplementedError

        if L < n_cluster:
            n_cluster = max(1, L)

        if L == 0:
            raise ValueError("No patches to cluster (L=0)")

        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(self.weight_params, dim=0)
        w_morph = weights[0]
        w_spatial = weights[1]

        if self.feature_weight == -1:
            labels = torch.randint(0, n_cluster, (L,), device=x.device, dtype=torch.long).cpu().numpy()
        else:
            with torch.no_grad():
                # Normalize features and coordinates on GPU
                feats = x.squeeze(0)
                coords_norm = coords.squeeze(0)
                feats_norm = (feats - feats.mean(dim=0)) / (feats.std(dim=0) + 1e-6)
                coords_norm = (coords_norm - coords_norm.mean(dim=0)) / (coords_norm.std(dim=0) + 1e-6)
                feats_norm = F.normalize(feats_norm, p=2, dim=1)

                # Batch processing for similarity computation
                batch_size = 1000
                k = min(10, L - 1)
                indices = torch.zeros(L, k, device=x.device, dtype=torch.long)
                values = torch.zeros(L, k, device=x.device, dtype=torch.float32)

                for i in range(0, L, batch_size):
                    end = min(i + batch_size, L)
                    feats_batch = feats_norm[i:end]
                    coords_batch = coords_norm[i:end]

                    # Cosine similarity
                    cosine_sim = torch.matmul(feats_batch, feats_norm.T)

                    # Euclidean distance
                    dist = torch.cdist(coords_batch, coords_norm, p=2)
                    sigma = dist.std() + 1e-6
                    spatial_sim = torch.exp(-dist / sigma)

                    # Combined similarities
                    similarity = w_morph * cosine_sim + w_spatial * spatial_sim
                    del cosine_sim, dist, spatial_sim

                    # Top-k neighbors
                    topk_vals, topk_idx = torch.topk(similarity, k, dim=1)
                    indices[i:end] = topk_idx
                    values[i:end] = topk_vals
                    del similarity, topk_vals, topk_idx

                # Sparse k-NN graph
                knn_graph = values / (values.sum(dim=1, keepdim=True) + 1e-6)

                # cupy for cuml.KMeans
                knn_graph_cupy = cp.asarray(knn_graph)

                try:
                    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', tol=1e-4, max_iter=50, random_state=1)
                    labels = kmeans.fit_predict(knn_graph_cupy)
                    labels = cp.asnumpy(labels)

                except Exception as e:
                    print(f"Graph-based clustering failed: {e}, falling back to random clustering")
                    labels = np.random.randint(0, n_cluster, size=L)
                    print(f"Random labels: {labels[:10]}...")

                torch.cuda.empty_cache()

        if not (labels >= 0).all() or not (labels < n_cluster).all():
            raise ValueError(f"Invalid labels: min={labels.min()}, max={labels.max()}, n_cluster={n_cluster}")

        index = np.argsort(labels, kind='stable').tolist()
        if max(index, default=-1) >= L or min(index, default=L) < 0:
            raise ValueError(f"Invalid index values: min={min(index)}, max={max(index)}, L={L}")

        window_sizes = np.bincount(labels).tolist()
        window_sizes_new = []
        for size in window_sizes:
            if size >= (self.cluster_size * 2 if self.cluster_size else 2):
                num_splits = max(1, size // (self.cluster_size or 1))
                quotient = size // num_splits
                remainder = size % num_splits
                result = [quotient + 1 if i < remainder else quotient for i in range(num_splits)]
                window_sizes_new.extend(result)
            else:
                window_sizes_new.append(size)
        window_sizes = window_sizes_new

        clusters = []
        coord_clusters = []
        index_tensor = torch.tensor(index, device=x.device, dtype=torch.long)
        x = x[:, index_tensor]
        coords = coords[:, index_tensor]

        now = 0
        for size in window_sizes:
            if size > 0:
                clusters.append(x[:, now:now + size])
                coord_clusters.append(coords[:, now:now + size])
                now += size

        if not clusters:
            raise ValueError("No valid clusters formed")
        if sum(window_sizes) != L:
            raise ValueError(f"Sum of window sizes {sum(window_sizes)} does not match L={L}")

        # `index` maps: position in the concatenated/clustered sequence -> position in
        # the sequence that was PASSED IN to this call. Callers must compose this across
        # layers/branches to recover the mapping back to the original WSI patch order.
        return clusters, coord_clusters, labels, index

    def forward(self, x, coords=None, cluster_label=None, return_patch_label=False):
        B, L, C = x.shape

        if cluster_label is not None:
            labels = cluster_label.numpy()[0]
            clusters, _, _, index = self.cluster_patches(x, coords)
        else:
            clusters, _, labels, index = self.cluster_patches(x, coords)

        refined_clusters = []
        for L_i in clusters:
            L_i_prime = self.local_attention(L_i)
            refined_clusters.append(L_i_prime)

        cluster_reps = []
        for L_i_prime in refined_clusters:
            R_i = torch.mean(L_i_prime, dim=1)
            cluster_reps.append(R_i)

        R = torch.stack(cluster_reps, dim=1)
        R_prime = self.global_attention(R)
        h = torch.cat(refined_clusters, dim=1)
        R_prime_expanded = R_prime.mean(dim=1, keepdim=True).expand(-1, h.shape[1], -1)
        h = h + R_prime_expanded

        index_t = torch.as_tensor(index, device=x.device, dtype=torch.long)

        if return_patch_label:
            return h, labels, index_t
        return h, None, index_t


class SGCMLL(nn.Module):
    def __init__(self, n_classes=4, input_size=384, hidden_size=None, deep=1, n_cluster=None, cluster_size=None,
                 feature_weight=0, as_backbone=False, dropout_rate=0.25, dynamic_filter=False, use_filter_branch=False,
                 with_clusatten=True, dynamicfilter_quantile=0.25, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.deep = deep
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        self.as_backbone = as_backbone
        self.dynamic_filter = dynamic_filter
        self.with_clusatten = with_clusatten
        self.dynamicfilter_quantile = dynamicfilter_quantile
        if self.dynamic_filter:
            self.use_filter_branch = True
        else:
            self.use_filter_branch = use_filter_branch

        if self.dynamic_filter:
            self.dynamicfilter = DynamicFilterQuant(dim=input_size, hidden_size=256, deep=1, quantile=dynamicfilter_quantile)

        if hidden_size is None:
            hidden_size = input_size
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        if with_clusatten:
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
        B, L_in, _ = x.shape

        patch_order = torch.arange(L_in, device=x.device, dtype=torch.long)

        filter_logits = None
        idx_high = idx_low = None
        if self.dynamic_filter:
            x, filter_logits, idx_high, idx_low, threshold = self.dynamicfilter(x)
            if return_IS:
                res['IS'] = filter_logits.squeeze().cpu().detach().numpy()
                res['threshold'] = threshold.item()

        if self.use_filter_branch:
            if idx_high is None:
                raise RuntimeError(
                    "use_filter_branch=True requires dynamic_filter=True to produce idx_high/idx_low."
                )
            h = x[:, idx_high]
            coords_filtered = coords[:, idx_high] if coords is not None else None
            h_app = x[:, idx_low]
            high_order = idx_high  # original indices of the patches in the clustered branch
            low_order = idx_low    # original indices of the patches bypassing clustering
        else:
            h = self.fc1(x)
            coords_filtered = coords
            high_order = patch_order

        patch_label = None
        if self.with_clusatten:
            if h.shape[1] > 1:
                for atten in self.attens:
                    h, patch_label, perm = atten(h, coords_filtered, cluster_label, return_patch_label)
                    high_order = high_order[perm]
            else:
                patch_label = None

        if return_patch_label:
            total_len = len(filter_logits.squeeze()) if self.use_filter_branch else h.shape[1]
            patch_label_all = np.zeros(total_len) - 1
            if h.shape[1] > 1:
                if self.use_filter_branch:
                    patch_label_all[idx_high.cpu().numpy()] = patch_label
                else:
                    patch_label_all = patch_label
            res['patch_label'] = patch_label_all
            return res

        if return_IS:
            return res

        if self.use_filter_branch:
            h = torch.cat([h, h_app], dim=1)
            full_order = torch.cat([high_order, low_order], dim=0)
        else:
            full_order = high_order

        if self.as_backbone:
            
            return {'feat': h, 'patch_order': full_order}

        h_pooled, att, attn_logits = self.amil(h)
        logits = self.classifier(h_pooled)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        res = {
            'logits': logits,
            'y_hat': Y_hat,
            'y_prob': Y_prob,
            'att': att,
            'patch_order': full_order,
        }
        return res
