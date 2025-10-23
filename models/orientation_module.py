import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class OrientNet(nn.Module):
    """Orientation estimation network copied from SE-ORNet."""

    def __init__(
        self,
        input_dims=None,
        output_dim=None,
        latent_dim=None,
        mlps=None,
        num_neighs=24,
        input_neighs=27,
        num_class=8,
    ):
        super().__init__()
        if input_dims is None:
            input_dims = [3, 64, 128, 256]
        if mlps is None:
            mlps = [256, 128, 128]

        self.num_neighs = num_neighs
        self.input_neighs = input_neighs

        modules = []
        in_dim = input_dims[0]
        for next_dim in input_dims[1:]:
            modules.append(EdgeConvModule(self.input_neighs, in_dim, next_dim))
            in_dim = next_dim
        self.input_modules = nn.ModuleList(modules)

        if latent_dim is None:
            latent_dim = in_dim
        if output_dim is None:
            output_dim = in_dim

        self.orient_module = OrientModule(self.num_neighs, in_dim, latent_dim)
        self.edgeconv = EdgeConvModule(self.num_neighs, latent_dim, output_dim)

        mlp_layers = []
        mlp_input_dim = 2 * (latent_dim + output_dim)
        for dim in mlps:
            mlp_layers.append(
                nn.Sequential(
                    nn.Conv1d(mlp_input_dim, dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            mlp_input_dim = dim
        self.mlps = nn.ModuleList(mlp_layers)

        self.num_class = num_class
        self.classifier = nn.Conv1d(mlp_input_dim, self.num_class, 1, bias=False)

        self.global_netD1 = nn.Sequential(
            nn.Conv1d(2 * (latent_dim + output_dim), 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.global_netD2 = nn.Linear(128, 2)

    def forward(self, xyz_s, xyz_t):
        batch_size = xyz_s.shape[0]
        idx_s = knn(xyz_s, xyz_s, k=self.input_neighs)
        idx_t = knn(xyz_t, xyz_t, k=self.input_neighs)
        feature_s = xyz_s.transpose(1, 2)
        feature_t = xyz_t.transpose(1, 2)

        for module in self.input_modules:
            feature_s = module(feature_s, idx=idx_s)
            feature_t = module(feature_t, idx=idx_t)

        latent_s_0 = self.orient_module(xyz_s, xyz_t, feature_s, feature_t)
        latent_t_0 = self.orient_module(xyz_t, xyz_s, feature_t, feature_s)
        latent_s_1 = self.edgeconv(latent_s_0)
        latent_t_1 = self.edgeconv(latent_t_0)

        x = torch.cat((latent_s_0, latent_s_1), dim=1)
        y = torch.cat((latent_t_0, latent_t_1), dim=1)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)

        x = torch.cat((x1, x2), 1).unsqueeze(-1)
        y = torch.cat((y1, y2), 1).unsqueeze(-1)

        d_input = torch.cat((x, y), dim=-1)
        global_d_pred = self.global_netD1(grad_reverse(d_input))
        global_d_pred = torch.mean(global_d_pred, dim=2)
        global_d_pred = self.global_netD2(global_d_pred)

        for mlp in self.mlps:
            x = mlp(x)
            y = mlp(y)

        angle_x = self.classifier(x).squeeze(-1)
        angle_y = self.classifier(y).squeeze(-1)

        return {
            "angle_x": angle_x,
            "angle_y": angle_y,
            "global_d_pred": global_d_pred,
        }


class OrientModule(nn.Module):
    def __init__(self, k, input_size, output_size):
        super().__init__()
        self.k = k
        self.linear = nn.Conv2d((input_size + 3) * 2, output_size, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, xyz_s, xyz_t, feature_s, feature_t):
        feature, xyz = get_graph_feature(
            feature_s.transpose(2, 1),
            feature_t.transpose(2, 1),
            k=self.k,
            ref_xyz=xyz_s,
        )
        xyz_t = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)
        feature_t = feature_t.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - feature_t, feature, xyz - xyz_t, xyz_t), dim=-1)
        feature = feature.permute(0, 3, 1, 2).contiguous()
        feature = self.relu(self.bn(self.linear(feature)))
        output, _ = feature.max(dim=-1, keepdim=False)
        return output


class EdgeConvModule(nn.Module):
    def __init__(self, k, input_size, output_size):
        super().__init__()
        self.k = k
        self.linear = nn.Conv2d(input_size * 2, output_size, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(output_size)

    def forward(self, inputs, idx=None):
        feature = get_graph_feature(inputs.transpose(2, 1), inputs.transpose(2, 1), k=self.k, idx=idx)
        inputs = inputs.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - inputs, inputs), dim=-1)
        feature = feature.permute(0, 3, 1, 2).contiguous()
        feature = self.relu(self.bn(self.linear(feature)))
        output, _ = feature.max(dim=-1, keepdim=False)
        return output


def knn(ref, query, k):
    if torch.cuda.is_available():
        try:
            from knn_cuda import KNN  # type: ignore

            _, idx = KNN(k=k, transpose_mode=True)(ref, query)
            return idx
        except Exception:
            pass

    inner = -2 * torch.matmul(ref, query.transpose(2, 1))
    xx = torch.sum(ref.transpose(2, 1) ** 2, dim=1, keepdim=True)
    yy = torch.sum(query.transpose(2, 1) ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(ref, query, k=20, idx=None, ref_xyz=None):
    batch_size, num_points, num_dims = ref.size()
    if idx is None:
        idx = knn(ref, query, k=k)
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    feature = ref.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    if ref_xyz is not None:
        xyz = ref_xyz.reshape(batch_size * num_points, -1)[idx, :]
        xyz = xyz.view(batch_size, num_points, k, 3)
        return feature, xyz
    return feature


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)
