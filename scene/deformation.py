"""Self-Adaptive Weighted Deformation Decoder (SAD).

Implements the attention-driven, dynamically weighted 4D deformation decoder
proposed in SAGS. The decoder uses multi-head self-attention combined with
feed-forward MLPs, gated by learnable weights (gamma_1, gamma_2) and Affine
transformations (alpha, beta), to capture both global geometric consistency
and local non-rigid tissue deformations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange
from scene.hexplane import HexPlaneField


class Attention(nn.Module):
    """Multi-head self-attention (Eq. 5 in the paper)."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Aff(nn.Module):
    """Learnable Affine transformation: Affine(x) = alpha * x + beta."""

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        return x * self.alpha + self.beta


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SADBlock(nn.Module):
    """Self-Adaptive Deformation Block (Eqs. 6-7 in the paper).

    y' = Affine_pre(x) + gamma_1 * MSA(Affine_pre(x))
    y  = Affine_post(y') + gamma_2 * MLP(Affine_post(y')) + x
    """

    def __init__(self, dim, mlp_dim, dropout=0., init_values=1e-4):
        super().__init__()
        self.pre_affine = Aff(dim)
        self.post_affine = Aff(dim)
        self.attention = Attention(dim, heads=4, dim_head=32)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        residual = x
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.attention(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x + residual


class Deformation(nn.Module):
    """4D Deformation field using HexPlane encoding + SAD blocks.

    Encodes spatio-temporal Gaussian information via HexPlane, processes it
    through stacked SAD blocks, and outputs deformation residuals for
    position, scale, rotation, and opacity.
    """

    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.no_grid = args.no_grid
        self.args = args

        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self._build_output_heads()

    def _build_output_heads(self):
        input_channels = self.grid.feat_dim if not self.no_grid else 4
        layers = [nn.Linear(input_channels, self.W), nn.GELU()]

        for _ in range(self.D - 1):
            layers.append(nn.GELU())
            layers.append(SADBlock(self.W, self.W))

        self.feature_out = nn.Sequential(*layers)

        pos_head = nn.Sequential(nn.GELU(), nn.Linear(self.W, self.W), nn.GELU(), nn.Linear(self.W, 3))
        scale_head = nn.Sequential(nn.GELU(), nn.Linear(self.W, self.W), nn.GELU(), nn.Linear(self.W, 3))
        rot_head = nn.Sequential(nn.GELU(), nn.Linear(self.W, self.W), nn.GELU(), nn.Linear(self.W, 4))
        opacity_head = nn.Sequential(nn.GELU(), nn.Linear(self.W, self.W), nn.GELU(), nn.Linear(self.W, 1))
        return pos_head, scale_head, rot_head, opacity_head

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:, :3], time_emb[:, :1]], -1)
        else:
            grid_feature = self.grid(rays_pts_emb[:, :3], time_emb[:, :1])
            h = grid_feature

        h = h.unsqueeze(1)
        h = self.feature_out(h).squeeze(1)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:, :3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self, rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()

        # Compute deformation residuals (Eqs. 8-9)
        pts = rays_pts_emb[:, :3] if self.args.no_dx else rays_pts_emb[:, :3] + self.pos_deform(hidden)
        scales = scales_emb[:, :3] if self.args.no_ds else scales_emb[:, :3] + self.scales_deform(hidden)
        rotations = rotations_emb[:, :4] if self.args.no_dr else rotations_emb[:, :4] + self.rotations_deform(hidden)
        opacity = opacity_emb[:, :1] if self.args.no_do else opacity_emb[:, :1] + self.opacity_deform(hidden)

        return pts, scales, rotations, opacity

    def get_mlp_parameters(self):
        return [p for name, p in self.named_parameters() if "grid" not in name]

    def get_grid_parameters(self):
        return list(self.grid.parameters())


class deform_network(nn.Module):
    """Top-level deformation network wrapping time encoding + Deformation."""

    def __init__(self, args):
        super().__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output

        times_ch = 2 * timebase_pe + 1
        self.timenet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(),
            nn.Linear(timenet_width, timenet_output))

        self.deformation_net = Deformation(
            W=net_width, D=defor_depth,
            input_ch=(4 + 3) + ((4 + 3) * scale_rotation_pe) * 2,
            input_ch_time=timenet_output, args=args)

        self.register_buffer('time_poc', torch.FloatTensor([(2 ** i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2 ** i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2 ** i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2 ** i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)

    def forward_static(self, points):
        return self.deformation_net(points)

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        means3D, scales, rotations, opacity = self.deformation_net(
            point, scales, rotations, opacity, times_sel)
        return means3D, scales, rotations, opacity

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight, gain=1)
