import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from graph.mediapipe_graph import Graph


class STGCN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_class=3,
        edge_importance_weighting=True,
        dropout=0.5,
    ):

        super().__init__()

        self.graph = Graph()

        A = torch.tensor(
            self.graph.A,
            dtype=torch.float32,
            requires_grad=False,
        )

        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9

        kernel_size = (
            temporal_kernel_size,
            spatial_kernel_size,
        )

        self.data_bn = nn.BatchNorm1d(
            in_channels * self.graph.num_node
        )

        kwargs = dict(dropout=dropout)

        self.st_gcn_networks = nn.ModuleList(
            (
                STGCNBlock(
                    in_channels,
                    64,
                    kernel_size,
                    1,
                    residual=False,
                ),

                STGCNBlock(
                    64,
                    64,
                    kernel_size,
                    1,
                    **kwargs
                ),

                STGCNBlock(
                    64,
                    64,
                    kernel_size,
                    1,
                    **kwargs
                ),

                STGCNBlock(
                    64,
                    64,
                    kernel_size,
                    1,
                    **kwargs
                ),