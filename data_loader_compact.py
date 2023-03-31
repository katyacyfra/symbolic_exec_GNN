import argparse
import json
import os.path
from torch_geometric.data import Data, HeteroData
from os import walk
from typing import Dict, Tuple
import torch
import numpy as np
from operator import itemgetter

from game_compact import GameState

#NUM_NODE_FEATURES = 49
NUM_NODE_FEATURES = 6
EXPECTED_FILENAME = "expectedResults.txt"
GAMESUFFIX = "_gameState"
STATESUFFIX = "_statesInfo"


class ServerDataloaderHeteroVector():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph_types_and_data = {}
        self.dataset = []
        self.process_directory(data_dir)
        self.__process_files()

    @staticmethod
    def convert_input_to_tensor(input: GameState) -> Tuple[HeteroData, Dict[int, int]]:
        """
        Converts game env to tensors
        """
        graphVertices = input.GraphVertices
        game_states = input.States
        game_edges = input.Map
        data = HeteroData()
        nodes_vertex_set = set()
        nodes_state_set = set()
        nodes_vertex = []
        nodes_state = []
        edges_index_v_v = []
        edges_index_s_s = []
        edges_index_s_v_in = []
        edges_index_v_s_in = []
        edges_index_s_v_history = []
        edges_index_v_s_history = []
        edges_attr_v_v = []

        state_map: Dict[int, int] = {}  # Maps real state id to its index
        vertex_map: Dict[int, int] = {}  # Maps real vertex id to its index
        vertex_index = 0
        state_index = 0

        # vertex nodes
        for v in graphVertices:
            vertex_id = v.Id
            if vertex_id not in vertex_map:
               vertex_map[vertex_id] = vertex_index # maintain order in tensors
               vertex_index = vertex_index + 1
               nodes_vertex.append(np.array([int(v.InCoverageZone),
                                          v.BasicBlockSize, int(v.CoveredByTest),
                                          int(v.VisitedByState), int(v.TouchedByState)
                                          ]))
        # vertex -> vertex edges
        for e in game_edges:
            edges_index_v_v.append(np.array([vertex_map[e.VertexFrom],
                                             vertex_map[e.VertexTo]]))
            edges_attr_v_v.append(np.array([e.Label.Token])) #TODO: consider token in a model

        # state nodes
        for s in game_states:
            state_id = s.Id
            if state_id not in state_map:
                state_map[state_id] = state_index
                nodes_state.append(np.array([s.Position,
                                             s.PredictedUsefulness, s.PathConditionSize,
                                             s.VisitedAgainVertices,
                                             s.VisitedNotCoveredVerticesInZone,
                                             s.VisitedNotCoveredVerticesOutOfZone
                                             ]))
            # history edges: state -> vertex and back
                for h in s.History: #TODO: process NumOfVisits as edge label
                   v_to = vertex_map[h.GraphVertexId]
                   edges_index_s_v_history.append(np.array([state_index, v_to]))
                   edges_index_v_s_history.append(np.array([v_to, state_index]))
                state_index = state_index + 1

        # state and its childen edges: state -> state
        for s in game_states:
            for ch in s.Children:
                edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))


        # state position edges: vertex -> state and back
        for v in graphVertices:
            for s in v.States:
                edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
                edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

        data['game_vertex'].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
        data['state_vertex'].x = torch.tensor(np.array(nodes_state), dtype=torch.float)
        data['game_vertex', 'to', 'game_vertex'].edge_index = torch.tensor(np.array(edges_index_v_v),
                                                                           dtype=torch.long).t().contiguous()
        data['state_vertex', 'in', 'game_vertex'].edge_index = torch.tensor(np.array(edges_index_s_v_in),
                                                                            dtype=torch.long).t().contiguous()
        data['game_vertex', 'in', 'state_vertex'].edge_index = torch.tensor(np.array(edges_index_v_s_in),
                                                                            dtype=torch.long).t().contiguous()
        data['state_vertex', 'history', 'game_vertex'].edge_index = torch.tensor(np.array(edges_index_s_v_history),
                                                                                 dtype=torch.long).t().contiguous()
        data['game_vertex', 'history', 'state_vertex'].edge_index = torch.tensor(np.array(edges_index_v_s_history),
                                                                                 dtype=torch.long).t().contiguous()
        if (edges_index_s_s):
            data['state_vertex', 'parent_of', 'state_vertex'].edge_index = torch.tensor(np.array(edges_index_s_s),
                                                                                        dtype=torch.long).t().contiguous()
            # print(data['state', 'parent_of', 'state'].edge_index)
        #data['game_vertex', 'to', 'game_vertex'].edge_attr = torch.tensor(np.array(edges_attr_v_v), dtype=torch.long)
        # data['state_vertex', 'to', 'game_vertex'].edge_attr = torch.tensor(np.array(edges_attr_s_v), dtype=torch.long)
        # data.state_map = state_map
        return data, state_map

    @staticmethod
    def get_expected_value(file_path: str, state_map: Dict[int, int]) -> torch.tensor:
        """Get tensor for states"""
        expected = {}
        with open(file_path) as f:
            data = json.load(f)
            state_set = set()
            for d in data:
                sid = d['StateId']
                if sid not in state_set:
                   state_set.add(sid)
                   values = list(d.values())
                   expected[values[0]] = np.array(values[1:])
        ordered = []
        ordered_by_index = list(zip(*sorted(state_map.items(), key=lambda x: x[1])))[0]
        for k in ordered_by_index:
            ordered.append(expected[k])
        return torch.tensor(np.array(ordered), dtype=torch.float)

    def process_directory(self, data_dir):
        example_dirs = next(walk(data_dir), (None, [], None))[1]
        example_dirs.sort()
        print(example_dirs)
        for fldr in example_dirs:
            fldr_path = os.path.join(data_dir, fldr)
            graphs_to_convert = []
            for f in os.listdir(fldr_path):
                if GAMESUFFIX in f:
                    graphs_to_convert.append(f)
            graphs_to_convert.sort(key=lambda x: int(x.split('_')[0]))
            self.graph_types_and_data[fldr] = graphs_to_convert

    def __process_files(self):
        for (k, v) in self.graph_types_and_data.items():
            for file in v:
                with open(os.path.join(self.data_dir, k, file)) as f:
                    print(os.path.join(self.data_dir, k, file))
                    data = json.load(f)
                    graph, state_map = self.convert_input_to_tensor(GameState.from_dict(data))
                    if graph is not None:
                        # add_expected values
                        expected = self.get_expected_value(os.path.join(self.data_dir, k,
                                                                        file.split("_")[0] + STATESUFFIX),
                                                           state_map)
                        #print(len(graph['state_vertex'].x), len(state_map), len(expected))
                        graph.y = expected
                        self.dataset.append(graph)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(prog='V# pytorch-geometric data conversion', description="Symbolic execution")
    parser.add_argument('--dataset', required=True, help="Dataset folder")
    parser.add_argument('--mode', help="heterogeneous or homogeneous graph model (het|hom)")


def get_data_hetero_vector():
    dl = ServerDataloaderHeteroVector("../../GNN_V#/Serialized_almost_all")
    # dl = ServerDataloaderHetero("../../GNN_V#/Serialized_test")
    return dl.dataset


if __name__ == '__main__':
    get_data_hetero_vector()