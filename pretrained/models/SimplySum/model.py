import torch
from torch.nn import Linear
from torch_geometric.nn import TAGConv, SAGEConv, GATConv, RGCNConv, ResGatedGraphConv, GraphConv
from torch_geometric.nn.conv.hetero_conv import group

class  StateModelEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = RGCNConv(7, hidden_channels, 3)
        self.conv12 = RGCNConv(hidden_channels, hidden_channels, 3)
        self.lin_vertices = Linear(7, hidden_channels)
        self.conv11 = TAGConv(hidden_channels, hidden_channels, 3)
        self.conv112 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = TAGConv(7, hidden_channels, 3)
        self.conv21 = TAGConv(7, hidden_channels, 3)
        self.conv212 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv((-1, -1), hidden_channels)
        #self.conv4 = GATConv((-1, -1), hidden_channels, edge_dim=2, add_self_loops=False)
        self.conv4 = GraphConv((-1, -1), hidden_channels)
        self.conv6 = GraphConv((-1, -1), hidden_channels)
        self.conv5 = GraphConv((-1, -1), hidden_channels)
        #self.conv5 = GATConv((-1, -1), hidden_channels, edge_dim=2, add_self_loops=False)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin7 = Linear(hidden_channels, out_channels)

    def forward(
            self,
            game_x,
            state_x,
            edge_index_v_v,
            edge_type_v_v,
            edge_index_history_v_s,
            edge_attr_history_v_s,
            edge_index_in_v_s,
            edge_index_s_s,
            edge_index_in_s_v,
            edge_index_history_s_v
    ):
        edge_attr_history_v_s = edge_attr_history_v_s[:, 0]

        if edge_type_v_v.numel() != 0:
            game_x = self.conv1(
                game_x,
                edge_index_v_v,
                edge_type_v_v
            ).relu()
        else:
            game_x = self.lin_vertices(game_x)

        game_x = self.conv11(
            game_x,
            edge_index_v_v,
        ).relu()

        game_x = self.conv3(
            (state_x, game_x),
            edge_index_in_s_v
        ).tanh()

        if edge_type_v_v.numel() != 0:
            game_x = self.conv12(
                game_x,
                edge_index_v_v,
                edge_type_v_v
            ).relu()

        game_x = self.conv11(
            game_x,
            edge_index_v_v,
        ).relu()

        game_x = self.conv4(
            (state_x, game_x),
            edge_index_history_s_v,
            edge_attr_history_v_s.float(),
        ).tanh()

        if edge_type_v_v.numel() != 0:
            game_x = self.conv12(
                game_x,
                edge_index_v_v,
                edge_type_v_v
            ).relu()

        game_x = self.conv11(
            game_x,
            edge_index_v_v,
        ).relu()

        state_x = self.conv2(
            state_x,
            edge_index_s_s,
        ).relu()

        state_x_hist = self.conv5(
            (game_x, state_x),
            edge_index_history_v_s,
            edge_attr_history_v_s,
        ).tanh()

        state_x_hist = self.lin2(state_x_hist).relu()

        state_x_in = self.conv6(
            (game_x, state_x),
            edge_index_in_v_s,
        ).tanh()

        state_x_in = self.lin7(state_x_in).relu()

        state_x = group([state_x_hist, state_x_in], "sum")
        return state_x