# src/models/model_architecture.py

"""
GAST: Graph-Augmented Spectral-Spatial Transformer for Hyperspectral Image Classification.
Spectral modeling is enhanced with Transformer encoder and positional embeddings.
Spatial modeling is performed via GATv2 on local patch graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError as e:
    raise ImportError("GAST requires PyTorch Geometric. ➜ pip install torch-geometric") from e


def _build_patch_edge_index(k: int, device: torch.device) -> torch.Tensor:
    if not hasattr(_build_patch_edge_index, "_cache"):
        _build_patch_edge_index._cache = {}
    if k in _build_patch_edge_index._cache:
        return _build_patch_edge_index._cache[k].to(device)

    grid = torch.arange(k * k).view(k, k)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    edges = []
    for i in range(k):
        for j in range(k):
            src = grid[i, j].item()
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < k and 0 <= nj < k:
                    edges.append([src, grid[ni, nj].item()])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    _build_patch_edge_index._cache[k] = edge_index
    return edge_index.to(device)


class GAST(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        patch_size: int = 11,
        spec_dim: int = 64,
        spat_dim: int = 64,
        n_heads: int = 8,
        n_layers: int = 4,
        transformer_heads: int = 8,      
        transformer_layers: int = 1,     
        dropout: float = 0.2,
        disable_spectral: bool = False,
        disable_spatial: bool = False,
        fusion_mode: str = "gate",  # gate | concat | spatial_only | spectral_only
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_nodes = patch_size ** 2
        self.disable_spectral = disable_spectral
        self.disable_spatial = disable_spatial
        self.fusion_mode = fusion_mode

        # Spectral branch: MLP + Transformer
        self.spectral_mlp = nn.Sequential(
            nn.Linear(in_channels, spec_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spec_dim, spec_dim)
        )

        # Positional embedding (learnable) for each node
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, self.n_nodes, spec_dim))
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)

        # Transformer encoder (1 layer, customizable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=spec_dim,
            nhead=transformer_heads,  # <-- use dynamic value
            dim_feedforward=spec_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.spectral_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers  # <-- use dynamic value
        )

        # Spatial branch: GATv2 layers
        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                spec_dim if i == 0 else spat_dim,
                spat_dim,
                heads=n_heads,
                concat=False,
                dropout=dropout
            )
            for i in range(n_layers)
        ])

        # Projection to match dims if needed
        self.proj = nn.Linear(spec_dim, spat_dim) if spec_dim != spat_dim else nn.Identity()

        # Fusion gate + classifier
        # self.gate = nn.Parameter(torch.tensor(0.5))  # in (0, 1)
        
        # Fusion gate: vector-based gating mechanism, learned from concatenated spectral and spatial features
        self.gate_mlp = None
        if fusion_mode == "gate":
            self.gate_mlp = nn.Sequential(
                nn.Linear(spat_dim * 2, spat_dim),
                nn.Sigmoid()
            )
        
        norm_in = spat_dim * 2 if fusion_mode == "concat" else spat_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(norm_in),
            nn.Linear(norm_in, spat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spat_dim * 2, n_classes)
        )

    def _spectral_encode(self, x: torch.Tensor) -> torch.Tensor:
        B, C, k, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, self.n_nodes, C)  # (B, N, C)
        h = self.spectral_mlp(x)  # (B, N, spec_dim)
        h = h + self.spectral_pos_embed  # positional encoding
        h = self.spectral_transformer(h)  # transformer encoder
        return h

    def _spatial_gat(self, h: torch.Tensor, device: torch.device) -> torch.Tensor:
        B, N, F = h.shape
        edge_index_template = _build_patch_edge_index(self.patch_size, device)
        E = edge_index_template.size(1)

        h = h.reshape(B * N, F)
        edge_index = edge_index_template.repeat(1, B)
        offsets = (torch.arange(B, device=device) * N).repeat_interleave(E)
        edge_index = edge_index + offsets.unsqueeze(0)

        for gat in self.gat_layers:
            h = gat(h, edge_index).relu_()

        return h.view(B, N, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        spec = None
        if not self.disable_spectral:
            spec = self._spectral_encode(x)

        spat = None
        if not self.disable_spatial:
            input_feat = spec if spec is not None else self._spectral_encode(x).detach()
            spat = self._spatial_gat(input_feat, device)

        if self.fusion_mode == "gate":
            if spec is None or spat is None:
                raise ValueError("Fusion mode 'gate' requires both spectral and spatial branches.")
            # fused = self.gate * spat + (1 - self.gate) * self.proj(spec)
            gate_input = torch.cat([self.proj(spec), spat], dim=-1)
            gate = self.gate_mlp(gate_input)
            fused = gate * spat + (1 - gate) * self.proj(spec)
     
        elif self.fusion_mode == "concat":
            if spec is None or spat is None:
                raise ValueError("Fusion mode 'concat' requires both branches.")
            fused = torch.cat([spat, self.proj(spec)], dim=-1)
        elif self.fusion_mode == "spatial_only":
            if spat is None:
                raise ValueError("Fusion mode 'spatial_only' requires spatial branch.")
            fused = spat
        elif self.fusion_mode == "spectral_only":
            if spec is None:
                raise ValueError("Fusion mode 'spectral_only' requires spectral branch.")
            fused = self.proj(spec)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        center = self.n_nodes // 2
        logits = self.classifier(fused[:, center, :])
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the fused feature vector for the center pixel (before classifier).
        """
        device = x.device

        spec = None
        if not self.disable_spectral:
            spec = self._spectral_encode(x)

        spat = None
        if not self.disable_spatial:
            input_feat = spec if spec is not None else self._spectral_encode(x).detach()
            spat = self._spatial_gat(input_feat, device)

        if self.fusion_mode == "gate":
            gate_input = torch.cat([self.proj(spec), spat], dim=-1)
            gate = self.gate_mlp(gate_input)
            fused = gate * spat + (1 - gate) * self.proj(spec)
        elif self.fusion_mode == "concat":
            fused = torch.cat([spat, self.proj(spec)], dim=-1)
        elif self.fusion_mode == "spatial_only":
            fused = spat
        elif self.fusion_mode == "spectral_only":
            fused = self.proj(spec)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        center = self.n_nodes // 2
        # Return the fused feature for the center pixel (no classifier)
        return fused[:, center, :]  # (batch, feature_dim)


__all__ = ["GAST"]

if __name__ == "__main__":
    import torch

    # === User Configurable Section ===
    used_tool = "torchviz"  # Options: 'torchviz', 'torchlens','torchview', 'tensorboard'
    save_dir = "./reports/figures/model_architecture"       # Directory to save visualizations (if applicable)
    input_shape = (2, 224, 11, 11)  # Example input shape (B, C, H, W)
    # ================================

    model = GAST(
        in_channels=224,
        n_classes=16,
        patch_size=11,
        spec_dim=64,
        spat_dim=64,
        n_heads=8,
        n_layers=4,
        transformer_heads=8,      
        transformer_layers=1,     
        dropout=0.2,
        disable_spectral=False,
        disable_spatial=False,
        fusion_mode="gate"
    )
    model.eval()
    x = torch.randn(*input_shape)