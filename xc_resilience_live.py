""" Log

copy for paper review (nature communications) May 2023

version 1.5 updated Jan 2023
unweighted network analysis for interconnected network
path_based_unweighted_relocation
ER-based null model (controlling edge length) benchmark

version 1.4 updated Nov 2022
GTFS compatible
read files from GTFS-style dataset
simple-paths-based betweenness
node visit frequency (OD flow analysis)
robustness_flow_weighted_target_list_attack (pre-determined target list)
add line_plot function (matplotlib)

Copy for paper publication (IEEE transactions on intelligent transportation systems)

version 1.3 updated 30 Sep 2021
travel distance calculation added
unweighted relocation analysis (node level)

version 1.2 updated
parameter trip_edge= {trip: [edge,...]}
(need manual read load_pet)
load edge list from trip_edge dict

version 1.1 updated
capacity-weighted model based on trips/routes
capacity-related parameters
combine two ClassResilience providing cross-layer edges
"""

import csv
import time
import numpy as np
import networkx as nx
import random
from collections import defaultdict
import copy
from multiprocessing import Pool
import matplotlib.pyplot as plt
import math
from networkx.algorithms import approximation as approx
from tqdm import tqdm
from itertools import permutations
from itertools import combinations
import toolbox
from toolbox import *
from NetworkStructure import Network


class Resilience:
    def __init__(self, graph_name, indexing=False):
        self.name = graph_name
        self.G = nx.DiGraph()  # graph model
        # geospatial data
        self.node_coordinates = {}  # (lat, lon)
        # flow-weighted model
        # self.flow_matrix = None
        self.od_flow = {}
        self.node_flow = {}
        self.node_flow_centrality = {}
        self.indexing = indexing
        if self.indexing:
            self.node2index = None
            self.index2node = None
        self._matrix_header = None  # load from adjacency matrix
        self._edge_dict = None
        self._relocation_edge_dict = None
        self._relocation_edge_weight = None
        self._restoration_edge_dict = None
        self._restoration_node_weight = None
        # capacity-weighted model based on trips/routes (GTFS data)
        self.network = Network(network_name=graph_name)  # network-route-trip structure
        # self.routes = {}  # system structure
        # self.stops = {}  # NOTE: standalone stop repository
        # self.route_edge = defaultdict(list)
        # self.edge_route = None
        # self.trip_param = defaultdict(dict)
        # self.node_param = defaultdict(dict)
        # self.edge_capacity = {}
        # self.node_capacity = {}
        # multi-processing
        self.core_num = 5

    # def _save_weights_to_json(self):
    #     res = {'gps': self.node_coordinates,
    #            'edge_trip': self.edge_trip,
    #            'trip_edge': self.trip_edge,
    #            'edge_param': self.edge_param,
    #            'node_param': self.node_param,
    #            'edge_capacity': self.edge_capacity,
    #            'node_capacity': self.node_capacity}
    #     save_pet(res, f'xc_{self.name}_weights')

    # def _load_weights_from_save(self):
    #     res = load_pet(f'xc_{self.name}_weights')
    #     self.node_coordinates = res['gps']
    #     self.edge_trip = res['edge_trip']
    #     self.trip_edge = res['trip_edge']
    #     self.edge_param = res['edge_param']
    #     self.node_param = res['node_param']
    #     self.edge_capacity = res['edge_capacity']
    #     self.node_capacity = res['node_capacity']

    def load_adjacency_matrix(self, file_path, contain_header=True):
        node_list = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            matrix = []
            for row in csv_reader:
                matrix.append(row)
        if contain_header:
            for i in range(len(matrix[0])):
                if matrix[0][i] != matrix[i][0]:
                    print('error: adjacency matrix has asymmetric headers')
            # delete headers
            for row in matrix:
                del row[0]
            header = matrix.pop(0)
            # use (network_name, node_name) represents node
            self._matrix_header = header
            node_list = [(self.name, node) for node in header]
        self.G.add_nodes_from(node_list)
        if self.indexing:
            self.node2index = {node: index for index, node in enumerate(node_list)}
            self.index2node = {index: node for index, node in enumerate(node_list)}
        for idx, x in enumerate(node_list):
            for idy, y in enumerate(node_list):
                if int(matrix[idx][idy]) > 0:
                    self.G.add_edge(x, y)
        print('\nnetwork created:',
              f'name = {self.name}, '
              f'number of nodes = {self.G.number_of_nodes()}, '
              f'number of edges = {self.G.number_of_edges()}')

    def load_edge_list(self, file_path, contain_header=True, u_col=1, v_col=2, route_col=None):
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            edge_list = []
            for row in csv_reader:
                edge_list.append(row)
        if contain_header:
            del edge_list[0]  # delete headers
        node_list = []
        for row in edge_list:
            node_list.extend([row[u_col], row[v_col]])
        node_list = [(self.name, node) for node in list(set(node_list))]
        self.G.add_nodes_from(node_list)
        '''
        if route_col is not None:
            for row in edge_list:
                u, v = (self.name, row[u_col]), (self.name, row[v_col])
                r = row[route_col]
                self.G.add_edge(u, v)
                self.route_edge[r].append((u, v))
            self.routes = list(self.route_edge.keys())
        else:
            for row in edge_list:
                u, v = (self.name, row[u_col]), (self.name, row[v_col])
                self.G.add_edge(u, v)
        print(f'{self.name}, '
              f'number of nodes = {self.G.number_of_nodes()}, '
              f'number of edges = {self.G.number_of_edges()}')
        '''

    # def print_dataset(self):
    #     print(f"|V|={self.G.number_of_nodes()}, |E|={self.G.number_of_edges()}")
    #     print(f"{'.routes':<20}{'list':<20}length={len(self.routes)}, example:{self.routes[0]}")
    #     print(f"{'.route_param':<20}{'dict of dict':<20}{self.route_param.keys()}")
    #     print(f"{'.node_param':<20}{'dict of dict':<20}{self.node_param.keys()}")
    #     print(f"{'.node_coordinates':<20}{'dict of tuple':<20}example:{next(iter(self.node_coordinates.items()))}")
    #     print(f"{'.trip_param':<20}{'dict of dict':<20}{self.trip_param.keys()}")
    #     print(f"{'.route_trip':<20}{'dict of list':<20}example:{next(iter(self.route_trip.items()))}")
    #     print(f"{'.trip_route':<20}{'dict':<20}example:{next(iter(self.trip_route.items()))}")
    #     print(f"{'.trip_edge':<20}{'dict of list':<20}example:{next(iter(self.trip_edge.items()))}")
    #     print(f"{'.edge_trip':<20}{'dict of list':<20}example:{next(iter(self.edge_trip.items()))}")

    # capacity-weighted model based on trips/routes (GTFS data)
    def update_graph_by_routes_data(self, intermodal_edge_list=None, update_node_label=False):
        self.G = nx.DiGraph()  # graph model
        for route in self.network.routes.values():
            agency_id = route.agency_id
            for trip in route.trips.values():
                edge_list = trip.edge_list()
                if edge_list:
                    self.G.add_edges_from(edge_list, label=agency_id)
                    # path = [edge[0] for edge in edge_list] + [edge_list[-1][-1]]
                    # nx.add_path(self.G, path, label=agency_id)
        for stop in self.network.stop_repository.values():
            self.node_coordinates[stop.stop_id] = (stop.stop_lat, stop.stop_lon)
        if intermodal_edge_list:
            self.G.add_edges_from(intermodal_edge_list)
        if update_node_label:
            for stop in self.network.stop_set():
                label = self.network.stop_repository[stop].label
                self.G.add_node(stop, label=label)

        # self.plot(show=True)
        # self.edge_trip = revert_dict_of_list(self.trip_edge)
        #
        # self.update_dataset()
        # self.print_dataset()
        #
        # if weight:
        #     weighted_network()
        # if walking_dst > 0:
        #     add_walkable_edges(dst_threshold=walking_dst)
        # # edges, weights = zip(*nx.get_edge_attributes(self.G, 'cap').items())
        # # print(np.max(list(weights)), np.min(list(weights)))
        # # self.plot(show=True, edge_color=weights, edge_cmap=plt.cm.Reds, edge_vmin=-20000)
        # # plot_distribution(list(weights))
        # for route in self.routes.values():
        #     if route.is_empty():
        #         route.show()

    '''
    def update_dataset(self):
        def data_cleaning():
            missing_route, missing_trip = [], []
            for rt in self.routes:
                if rt not in self.route_trip.keys():
                    missing_route.append(rt)
                else:
                    for tp in self.route_trip[rt]:
                        if tp not in self.trip_edge.keys():
                            missing_trip.append((rt, tp))
            if missing_route:
                print(f'data cleaning: removed route with missing data {missing_route}')
                for rt in missing_route:
                    self.routes.remove(rt)
            if missing_trip:
                print(f'data cleaning: removed trip with missing data {missing_trip}')
                for rt, tp in missing_trip:
                    self.route_trip[rt].remove(tp)

        def update_graph():
            self.G = nx.DiGraph()
            for trip, edge_list in self.trip_edge.items():
                agency_id = self.route_param['agency_id'][self.trip_route[trip]]
                path = [edge[0] for edge in edge_list] + [edge_list[-1][-1]]
                nx.add_path(self.G, path, label=agency_id, cap=0, cost=1)

        data_cleaning()
        update_graph()
        self.edge_trip = revert_dict_of_list(self.trip_edge)
        for key, item in revert_dict_of_list(self.route_trip).items():
            if len(item) == 1:
                self.trip_route[key] = item[0]
            else:
                print('improper trip data:', key, item)

    def load_edges_from_trip_edge(self):
        """
        use if self.trip_edge already defined
        self.trip_edge = {trip: list of edges}
        :return:
        """
        edge_pool = [edge for edge_seq in list(self.trip_edge.values())
                     for edge in edge_seq]
        self.G.add_edges_from(remove_duplicate(edge_pool))
        print(f'{self.name}, '
              f'number of nodes = {self.G.number_of_nodes()}, '
              f'number of edges = {self.G.number_of_edges()}')

    def load_edge_parameter(self, file_path, u_col=None, v_col=None, edge_col=None):
        """
        dict of dict format {(u,v):{param:value}}
        dataframe read csv; need header
        :param file_path: csv file
        :param u_col: column index for the first nodes of edges
        :param v_col: column index for the second nodes of edges
        :param edge_col: use it if edges are writen as (u, v) in single column
        :return:
        """
        if edge_col is not None:
            param = pd.read_csv(file_path, index_col=edge_col)
        else:
            param = pd.read_csv(file_path, index_col=[u_col, v_col])
        # eval("[]")
        for column in param.columns:
            for row in param.iterrows():
                param.loc[row[0], column] = eval(param.loc[row[0], column])
        param = param.to_dict(orient='index')
        self.edge_param = {edge: param[int(edge[0][1]), int(edge[1][1])]
                           for edge in self.get_edge_list()}

    def load_node_parameter(self, file_path, node_col):
        """
        dict of dict format {node:{param:value}}
        :param file_path: csv file
        :param node_col: column index for nodes' ids or names
        :return:
        """
        # dataframe read csv; need header.
        param = pd.read_csv(file_path, index_col=node_col).to_dict(orient='index')
        self.node_param = {node: param[node[1]]
                           for node in self.get_node_list()}

    def load_route_parameter(self, file_path, route_col, route_parent=False):
        params = pd.read_csv(file_path, dtype=str)
        index_name = params.columns[route_col]
        params.set_index(index_name, inplace=True)
        params = params.to_dict()
        sub_routes = defaultdict(list)
        if route_parent:  # dedicated for GMB dataset
            for r in self.routes:
                r_parent = r.split('-')[0]
                sub_routes[r_parent].append(r)
            for param_name, param in params.items():
                for r_parent, value in param.items():
                    r_children = sub_routes[str(r_parent)]
                    for r_child in r_children:
                        self.route_param[param_name][r_child] = value
        else:
            self.route_param = params
    '''

    def load_flow_matrix(self, file_path, contain_header=True):
        # same format as adjacency matrix
        # warning: must has same order of stations
        flow_matrix = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                flow_matrix.append(row)
        if contain_header:
            for i in range(1, len(flow_matrix[0])):
                if flow_matrix[0][i] != flow_matrix[i][0]:
                    print('Error: flow matrix has asymmetric headers')
                if flow_matrix[0][i].strip() != self.get_node_list()[i - 1][1].strip():
                    print(f'Warning: flow matrix has headers with unidentical station name: '
                          f'\"{flow_matrix[0][i]}\", \"{self.get_node_list()[i - 1][1]}\"')
                # if flow_matrix[0][i].strip() != self.matrix_header[i-1].strip():
                #     print(f'Warning: flow matrix has headers with unidentical station name: '
                #           f'\"{flow_matrix[0][i]}\", \"{self.matrix_header[i-1]}\"')
            for row in flow_matrix:
                del row[0]
            del flow_matrix[0]
        for idx in range(len(flow_matrix)):
            for idy in range(len(flow_matrix[idx])):
                if len(flow_matrix[idx][idy].strip()) < 1 or flow_matrix[idx][idy] == '/':
                    # print(f"data missing at position({x, y}), header excluded..")
                    flow_matrix[idx][idy] = 0.0
                else:
                    flow_matrix[idx][idy] = float(flow_matrix[idx][idy])
        for idx in range(len(flow_matrix)):
            for idy in range(len(flow_matrix[idx])):
                x, y = self.get_node_list()[idx], self.get_node_list()[idy]
                self.od_flow[x, y] = flow_matrix[idx][idy]

    def load_gps_coordinates(self, file_path, contain_header=True,
                             node_col=0, lat_col=1, lon_col=2):
        # # table: Node, Lat, Lon
        coordinates = []
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                coordinates.append(row)
        # print(coordinates)
        if contain_header:
            del coordinates[0]
        for row in coordinates:
            node, lat, lon = row[node_col], eval(row[lat_col]), eval(row[lon_col])
            if (self.name, node) in self.get_node_list():
                self.node_coordinates[(self.name, node)] = (lat, lon)
            # else:
            #     print(f'(load GPS) Error: station {(self.name, node)} not found in node list')

    def reachable_nodes(self, origin, edge_dict=None):
        current_node = origin
        visited = []
        path = [current_node]
        if edge_dict is None:
            edge_dict = self.get_edge_dict()
        while path:
            current_node = path.pop(0)
            visited.append(current_node)
            destinations = edge_dict[current_node]
            for next_node in destinations:
                if next_node not in visited and next_node not in path:
                    path.append(next_node)
        return visited

    def get_node_list(self):
        return list(self.G.nodes)

    def get_edge_list(self):
        return list(self.G.edges)

    def get_edge_dict(self, update=True):
        if update:
            edge_dict = defaultdict(list)
            for edge in self.G.edges():
                x, y = edge[0], edge[1]
                edge_dict[x].append(y)
            self._edge_dict = edge_dict
        return self._edge_dict

    def get_node_degree(self):
        return dict(nx.degree(self.G))

    def get_node_betweenness_centrality(self):
        return dict(nx.betweenness_centrality(self.G))

    def get_local_node_connectivity(self):
        od_pairs = permutations(self.get_node_list(), 2)
        return {od: approx.local_node_connectivity(G=self.G, source=od[0], target=od[1]) for od in od_pairs}

    def get_node_flow(self):
        if not self.od_flow:
            return None
        self.node_flow = {}
        for u in self.get_node_list():
            self.node_flow[u] = 0
        for u in self.get_node_list():
            # for v in self.reachable_nodes(u):
            for v in self.get_node_list():
                if v != u:  # avoid self-loop
                    self.node_flow[u] += self.od_flow[u, v]
                    self.node_flow[v] += self.od_flow[u, v]
        return self.node_flow

    def get_node_flow_centrality(self):
        node_list, queue = self.get_node_list(), self.get_node_list()
        if not self.od_flow:
            return None
        self.node_flow_centrality = {}
        total_flow = np.sum([value for value in self.od_flow.values()])
        sp = {}
        while queue:
            v = queue.pop(0)
            od_flow_v = 0
            for e1 in node_list:
                for e2 in node_list:
                    od_flow = self.od_flow[e1, e2]
                    if e1 != e2 and v not in [e1, e2] and od_flow > 0:
                        if (e1, e2) not in sp.keys():
                            try:
                                paths = list(nx.all_shortest_paths(self.G, e1, e2))
                                sp[e1, e2] = paths
                            except:
                                paths = None
                                sp[e1, e2] = paths
                        else:
                            paths = sp[e1, e2]
                        if paths:
                            nosp = len(paths)
                            for path in paths:
                                if v in path:
                                    od_flow_v += od_flow / nosp
            if total_flow == 0:
                self.node_flow_centrality[v] = 0
            else:
                self.node_flow_centrality[v] = od_flow_v / total_flow
        return self.node_flow_centrality

    def get_travel_distance(self, mean_value=False):
        spl = dict(nx.all_pairs_shortest_path_length(self.G))
        travel_distance_distribution = {}
        for od_pair, flow in self.od_flow.items():
            trip_len = spl[od_pair[0]][od_pair[1]]
            if trip_len in travel_distance_distribution.keys():
                travel_distance_distribution[trip_len] += flow
            else:
                travel_distance_distribution[trip_len] = flow
        # print(travel_distance_distribution)
        if mean_value:
            total_trip_len, total_flow = 0.0, 0.0
            for trip_len, flow in travel_distance_distribution.items():
                total_trip_len += trip_len * flow
                total_flow += flow
            mean_trip_len = total_trip_len / total_flow
            return round(mean_trip_len, 3)
        else:
            return travel_distance_distribution

    def get_travel_visit_freq(self):
        freq_all = {item: 0 for item in self.get_node_list()}
        total_flow = 0
        for od_pair, flow in self.od_flow.items():
            total_flow += flow
            s, t = od_pair
            all_simple_paths = list(nx.all_simple_paths(self.G, s, t))
            k = len(all_simple_paths)
            freq = defaultdict(lambda: 0)
            for path in all_simple_paths:
                for node in path:
                    freq[node] += flow
            for key, value in freq.items():
                freq_all[key] += value / k
        freq_all = {key: value / total_flow for key, value in freq_all.items()}
        return freq_all

    def get_node_sov(self, G):
        freq_all = {item: 0 for item in G.nodes()}
        ODs = list(combinations(G.nodes(), 2))
        K = len(ODs)
        if K < 1:
            return freq_all
        for s, t in ODs:
            all_simple_paths = list(nx.all_simple_paths(G, s, t))
            k = len(all_simple_paths)
            freq = defaultdict(lambda: 0)
            for path in all_simple_paths:
                for node in path:
                    freq[node] += 1
            for key, value in freq.items():
                freq_all[key] += value / k
        freq_all_nom = {key: value / K for key, value in freq_all.items()}
        return freq_all_nom

    def preparedness_node_degree_gini(self):
        node_degree = self.get_node_degree()
        return gini([value for value in node_degree.values()])

    def preparedness_node_betweenness_centrality_gini(self):
        node_betweenness_centrality = self.get_node_betweenness_centrality()
        return gini([value for value in node_betweenness_centrality.values()])

    def preparedness_node_flow_gini(self):
        if self.od_flow:
            node_flow = self.get_node_flow()
            return gini([value for value in node_flow.values()])
        else:
            return None

    def preparedness_node_flow_centrality_gini(self):
        if self.od_flow:
            node_flow_centrality = self.get_node_flow_centrality()
            return gini([value for value in node_flow_centrality.values()])
        else:
            return None

    def preparedness_gini(self):
        return {'Gini_ND': self.preparedness_node_degree_gini(),
                'Gini_BC': self.preparedness_node_betweenness_centrality_gini(),
                'Gini_NF': self.preparedness_node_flow_gini(),
                'Gini_FC': self.preparedness_node_flow_centrality_gini()}

    def robustness_unweighted_sov_based_attack(self, number_of_tests=100,
                                               multiple_removal=1,
                                               multi_processing=False):
        strategy, weight = 'sov', False
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def _unweighted_sequential_attack(self, strategy=None, multiple_removal=1):
        """
        single scenario test
        :param strategy: 'node_degree' or 'node_betweenness_centrality'
        :param multiple_removal: step size, increase it to reduce computational time
        :return: list of y value in a curve
        """
        temp_g = copy.deepcopy(self.G)
        node_list, n0 = list(temp_g.nodes()), temp_g.number_of_nodes()

        gcc = sorted(nx.strongly_connected_components(temp_g), key=len, reverse=True)
        n_g0 = temp_g.subgraph(gcc[0]).number_of_nodes()
        removed_list, degradation_curve = [], [n_g0 / n0]
        # print('n_g0 / n0 :', n_g0 / n0)
        step = 0
        while node_list:
            if strategy == 'node_degree':
                nd_dic = dict(nx.degree(temp_g))
                i = search_for_max(nd_dic, multiple_search=multiple_removal)
            elif strategy == 'node_betweenness_centrality':
                bc_dic = nx.betweenness_centrality(temp_g)
                i = search_for_max(bc_dic, multiple_search=multiple_removal)
            elif strategy == 'sov':
                sov_dic = self.get_node_sov(G=temp_g)
                i = search_for_max(sov_dic, multiple_search=multiple_removal)
            else:
                if multiple_removal > len(node_list):
                    i = copy.deepcopy(node_list)
                else:
                    i = random.sample(node_list, k=multiple_removal)
            if type(i) is list:
                removed_list.extend(i)
                temp_g.remove_nodes_from(i)
            else:
                removed_list.append(i)
                temp_g.remove_node(i)
            node_list = list(temp_g.nodes())
            step += 1
            if node_list:  # identify giant components
                gcc = sorted(nx.strongly_connected_components(temp_g), key=len, reverse=True)
                n_g0 = temp_g.subgraph(gcc[0]).number_of_nodes()
                degradation_curve.append(n_g0 / n0)
            else:
                degradation_curve.append(0)
        return degradation_curve

    def _flow_weighted_sequential_attack(self, strategy=None, multiple_removal=1):
        """
        single scenario test
        :param strategy: 'node_degree' or 'node_betweenness_centrality'
        :param multiple_removal: step size, increase it to reduce computational time
        :return: list of y value in a curve
        """
        temp = copy.deepcopy(self)
        node_list, n0 = list(temp.G.nodes()), temp.G.number_of_nodes()
        removed_list, degradation_curve = [], [1.0]
        step = 0
        total_flow = np.sum([value for value in self.od_flow.values()])
        while node_list:
            if strategy == 'node_degree':
                nd_dic = dict(nx.degree(temp.G))
                i = search_for_max(nd_dic, multiple_search=multiple_removal)
            elif strategy == 'node_betweenness_centrality':
                bc_dic = nx.betweenness_centrality(temp.G)
                i = search_for_max(bc_dic, multiple_search=multiple_removal)
            elif strategy == 'node_flow':
                nf_dic = temp.get_node_flow()
                i = search_for_max(nf_dic, multiple_search=multiple_removal)
            elif strategy == 'node_flow_centrality':
                fc_dic = temp.get_node_flow_centrality()
                i = search_for_max(fc_dic, multiple_search=multiple_removal)
            else:
                if multiple_removal > len(node_list):
                    i = copy.deepcopy(node_list)
                else:
                    i = random.sample(node_list, k=multiple_removal)
            if type(i) is list:
                removed_list.extend(i)
                temp.G.remove_nodes_from(i)
            else:
                removed_list.append(i)
                temp.G.remove_node(i)
            node_list = list(temp.G.nodes())
            step += 1
            if node_list:  # identify remaining flow
                remaining_flow = 0
                for origin in node_list:
                    reached = temp.reachable_nodes(origin)
                    for destination in reached:
                        remaining_flow += temp.od_flow[origin, destination]
                degradation_curve.append(remaining_flow / total_flow)
            else:
                degradation_curve.append(0)
        return degradation_curve

    def _robustness_repeated_tests(self, strategy, weight,
                                   number_of_tests=100,
                                   multiple_removal=1,
                                   multi_processing=False,
                                   save_raw_curves_to=None):
        # return degradation curve, format: [ys, xs]
        if multi_processing:
            curves = []
            pool = Pool(processes=self.core_num)
            if weight:
                pool_result = [pool.apply_async(self._flow_weighted_sequential_attack,
                                                args=(strategy, multiple_removal)) for test in
                               range(number_of_tests)]
            else:
                pool_result = [pool.apply_async(self._unweighted_sequential_attack,
                                                args=(strategy, multiple_removal)) for test in
                               range(number_of_tests)]
            pool.close()
            pool.join()
            for nt in pool_result:
                nt_curve = nt.get()
                curves.append(nt_curve)
            average_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            non = len(self.get_node_list())
            xs = [x / non for x in range(0, non, multiple_removal)]
            xs.append(1.0)
            print(f'rb_{strategy} =', numerical_integral_nml(average_curve, xs=xs))
        else:
            curves = []
            for test in tqdm(range(number_of_tests)):
                if weight:
                    curves.append(self._flow_weighted_sequential_attack(
                        strategy=strategy, multiple_removal=multiple_removal))
                else:
                    curves.append(self._unweighted_sequential_attack(
                        strategy=strategy, multiple_removal=multiple_removal))
            average_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            non = len(self.get_node_list())
            xs = [x / non for x in range(0, non, multiple_removal)]
            xs.append(1.0)
            print(f'rb_{strategy} =', numerical_integral_nml(average_curve, xs=xs))
        if save_raw_curves_to:
            toolbox.export_list(curves, filename=save_raw_curves_to)
        return [list(average_curve), xs, list(std_curve)]

    def robustness_unweighted_degree_based_attack(self, number_of_tests=100,
                                                  multiple_removal=1,
                                                  multi_processing=False):
        strategy, weight = 'node_degree', False
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_flow_weighted_degree_based_attack(self, number_of_tests=100,
                                                     multiple_removal=1,
                                                     multi_processing=False):
        strategy, weight = 'node_degree', True
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_unweighted_betweenness_based_attack(self, number_of_tests=100,
                                                       multiple_removal=1,
                                                       multi_processing=False):
        strategy, weight = 'node_betweenness_centrality', False
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_flow_weighted_betweenness_based_attack(self, number_of_tests=100,
                                                          multiple_removal=1,
                                                          multi_processing=False):
        strategy, weight = 'node_betweenness_centrality', True
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_unweighted_random_attack(self, number_of_tests=500,
                                            multiple_removal=1,
                                            multi_processing=False,
                                            save_raw_curves_to=None):
        strategy, weight = None, False
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing,
                                               save_raw_curves_to=save_raw_curves_to)

    def robustness_flow_weighted_random_attack(self, number_of_tests=1000,
                                               multiple_removal=1,
                                               multi_processing=False):
        strategy, weight = None, True
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_flow_weighted_node_flow_based_attack(self, number_of_tests=100,
                                                        multiple_removal=1,
                                                        multi_processing=False):
        strategy, weight = 'node_flow', True
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def robustness_flow_weighted_flow_centrality_based_attack(self, number_of_tests=100,
                                                              multiple_removal=1,
                                                              multi_processing=False):
        strategy, weight = 'node_flow_centrality', True
        return self._robustness_repeated_tests(strategy=strategy, weight=weight,
                                               number_of_tests=number_of_tests,
                                               multiple_removal=multiple_removal,
                                               multi_processing=multi_processing)

    def _robustness_flow_weighted_target_list_attack(self, strategy, multiple_removal=1, number_of_tests=1):
        curves = []
        for n_test in range(number_of_tests):
            # generate pre-determined target list
            non = self.G.number_of_nodes()
            if strategy == 'node_degree':
                nd_dic = dict(nx.degree(self.G))
                target_list = search_for_max(nd_dic, multiple_search=non)
            elif strategy == 'node_betweenness_centrality':
                bc_dic = nx.betweenness_centrality(self.G)
                target_list = search_for_max(bc_dic, multiple_search=non)
            elif strategy == 'node_flow':
                nf_dic = self.get_node_flow()
                target_list = search_for_max(nf_dic, multiple_search=non)
            elif strategy == 'node_flow_centrality':
                fc_dic = self.get_node_flow_centrality()
                target_list = search_for_max(fc_dic, multiple_search=non)
            else:
                target_list = []
                print('wrong parameter: strategy')
            if len(target_list) != non:
                print('error.................')
            temp = copy.deepcopy(self)
            node_list, n0 = list(temp.G.nodes()), temp.G.number_of_nodes()
            removed_list, degradation_curve = [], [1.0]
            step = 0
            total_flow = np.sum([value for value in self.od_flow.values()])
            while node_list:
                i, idi = [], 0
                while target_list and idi < multiple_removal:
                    i.append(target_list.pop(0))
                    idi += 1
                removed_list.extend(i)
                temp.G.remove_nodes_from(i)
                node_list = list(temp.G.nodes())
                step += 1
                if node_list:  # identify remaining flow
                    remaining_flow = 0
                    for origin in node_list:
                        reached = temp.reachable_nodes(origin)
                        for destination in reached:
                            remaining_flow += temp.od_flow[origin, destination]
                    degradation_curve.append(remaining_flow / total_flow)
                else:
                    degradation_curve.append(0)
            curves.append(degradation_curve)
        average_curve = np.mean(curves, axis=0)
        non = len(self.get_node_list())
        xs = [x / non for x in range(0, non, multiple_removal)]
        xs.append(1.0)
        print(f'(target list attack) rb_{strategy} =', numerical_integral_nml(average_curve, xs=xs))
        return average_curve, xs

    def _attack_sequence_generation(self, strategy, multiple_removal=1):
        temp = copy.deepcopy(self)
        node_list = copy.deepcopy(self.get_node_list())
        attack_sequence = []
        while node_list:
            if strategy == 'node_degree':
                nd_dic = dict(nx.degree(temp.G))
                i = search_for_max(nd_dic, multiple_search=multiple_removal)
            elif strategy == 'node_betweenness_centrality':
                bc_dic = nx.betweenness_centrality(temp.G)
                i = search_for_max(bc_dic, multiple_search=multiple_removal)
            elif strategy == 'node_flow':
                nf_dic = temp.get_node_flow()
                i = search_for_max(nf_dic, multiple_search=multiple_removal)
            elif strategy == 'node_flow_centrality':
                fc_dic = temp.get_node_flow_centrality()
                i = search_for_max(fc_dic, multiple_search=multiple_removal)
            else:
                if multiple_removal > len(node_list):
                    i = copy.deepcopy(node_list)
                else:
                    i = random.sample(node_list, k=multiple_removal)
            if type(i) is list:
                attack_sequence.extend(i)
                temp.G.remove_nodes_from(i)
            else:
                attack_sequence.append(i)
                temp.G.remove_node(i)
            node_list = temp.get_node_list()
        return attack_sequence

    def _relocation_dijsktra_weighted(self, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        edge_dict = self._relocation_edge_dict
        edge_weight = self._relocation_edge_weight
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()
        while current_node != end:
            visited.add(current_node)
            destinations = edge_dict[current_node]
            weight_to_current_node = shortest_paths[current_node][1]
            for next_node in destinations:
                weight = edge_weight[(current_node, next_node)] + weight_to_current_node
                # weight = 1 + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return math.inf
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        return shortest_paths[end][1]

    def _haversine_distance_between_nodes(self, node_x, node_y):
        lat1, lon1 = self.node_coordinates[node_x]
        lat2, lon2 = self.node_coordinates[node_y]
        return haversine(lat1, lon1, lat2, lon2)

    def unweighted_relocation(self, d_max=1600):
        """
        :return: dict of node relocation
        """
        def relo(l_list):
            lt = sum(l_list)
            cr_list = [lt / l for l in l_list]
            ct = sum(cr_list)
            cn_list = [cr / ct for cr in cr_list]
            return sum([cn * df(l_list[i]) for i, cn in enumerate(cn_list)])
        def df(d):
            will = 1 - d / d_max
            if 0 <= will <= 1:
                return will
            else:
                return 0
        print(len(self.get_node_list()))
        od_pairs = permutations(self.get_node_list(), 2)
        distance_matrix = {od: self._haversine_distance_between_nodes(od[0], od[1]) for od in od_pairs}
        # relocation_matrix = {od: df(dst) for od, dst in distance_matrix.items()}
        relocation_potential = {}
        for node in self.get_node_list():
            dsts = []
            for other in self.get_node_list():
                if node != other:
                    dst = distance_matrix[node, other]
                    if 1 <= dst < d_max:
                        dsts.append(dst)
                    elif 0 <= dst < 1:
                        dsts.append(1.0)  # tolerance
                    else:
                        pass
            if dsts:
                # relocation_potential[node] = df(harmonic_mean(dsts))
                relocation_potential[node] = relo(dsts)
            else:
                relocation_potential[node] = 0
        return relocation_potential

    def path_based_unweighted_relocation(self, d_max=1600):
        from toolbox import multiprocess_function
        def df(d):
            will = 1 - d / d_max
            if 0 <= will <= 1:
                return will
            else:
                return 0

        def reachable_nodes(G, origin):
            edge_dict = defaultdict(list)
            for u, v in list(G.edges):
                edge_dict[u].append(v)
            current_node = origin
            visited = []
            queue = [current_node]
            while queue:
                current_node = queue.pop(0)
                visited.append(current_node)
                destinations = edge_dict[current_node]
                for next_node in destinations:
                    if next_node not in visited and next_node not in queue:
                        queue.append(next_node)
            return visited

        G = self.G
        nodes = list(G.nodes)
        od_pairs = [(u, v) for u in nodes for v in nodes]
        distance_matrix = {od: self._haversine_distance_between_nodes(od[0], od[1]) for od in od_pairs}
        relocation_potential = {}

        for u in nodes:
            # time1 = time.time()
            reaches = set(reachable_nodes(G, origin=u))
            denom = len(reaches)
            length = denom
            dG = copy.deepcopy(G)
            dG.remove_node(u)
            if denom != 0:
                neighbors = {v: distance_matrix[u, v] for v in nodes if u != v and 0 <= distance_matrix[u, v] < d_max}
                neighbors = dict(sorted(neighbors.items(), key=lambda item: item[1]))
                relocation = 0
                # time2 = time.time()
                # while reaches:
                #     neighbors
                for v, dst in neighbors.items():
                    # new_reaches = set(reachable_nodes(dG, origin=v))
                    new_reaches = set(nx.descendants(dG, source=v))
                    reaches -= new_reaches
                    relocation += df(dst) * (length - len(reaches))
                    length = len(reaches)
                    if not reaches:
                        break
                relocation_potential[u] = relocation / denom
            else:
                relocation_potential[u] = 0
            # time_e = time.time()
            # print(f'{u}: {time_e - time2}/{time_e-time1}')
            # print(f'{u}:{round(relocation_potential[u], 3)}')
        return relocation_potential

    def _flow_weighted_relocation(self, v_a, v_b, cum_dst_prev=None):
        """
        used in recovery()
        functions: distance(), dijsktra_weighted(), df().
        :return:
        """

        def df(d):
            will = 1 - d / 1600
            if 0 <= will <= 1:
                return will
            else:
                return 0

        cum_dst = {}  # cumulative walking distance
        if cum_dst_prev is not None:
            for od_pair, cd_value in cum_dst_prev.items():
                # in restoration scenario, we keep value for intact/recovered path
                if cd_value == 0:
                    cum_dst[od_pair] = 0
        self._relocation_edge_dict = defaultdict(list)  # edge_dict[current_node]=next_nodes`
        self._relocation_edge_weight = {}  # edge_weight[node1,node2]=walking_distance
        max_flows = np.sum([value for value in self.od_flow.values()])
        for x in v_a:
            for y in v_a:
                if y in self._edge_dict[x]:  # connected by operational metro line
                    self._relocation_edge_dict[x].append(y)
                    self._relocation_edge_weight[x, y] = 0  # no walking distance
            for z in v_b:
                dst = self._haversine_distance_between_nodes(x, z)  # walking distance
                if dst <= 1600:  # introduce a relocation edge if no farther than 1600 metres
                    self._relocation_edge_dict[x].append(z)
                    self._relocation_edge_dict[z].append(x)
                    self._relocation_edge_weight[x, z] = dst
                    self._relocation_edge_weight[z, x] = dst
        total_relocation = 0
        for od_pair, flow in self.od_flow.items():
            # print('Debug: cd =', cd)
            if od_pair not in cum_dst.keys():  # cd=cumulative walking distance for the shortest path
                cum_dst[od_pair] = self._relocation_dijsktra_weighted(od_pair[0], od_pair[1])
            elif cum_dst[od_pair] != 0:  # keep the 0 value for intact/recovered path
                # if not 0 value, recalculate: (inf) previously disconnected or (>0) need walking
                cum_dst[od_pair] = self._relocation_dijsktra_weighted(od_pair[0], od_pair[1])
            if cum_dst[od_pair] > 0:
                will = df(cum_dst[od_pair])  # relocation rate based on cd
                relocation = will * flow  # relocated flow for the od_pair
                total_relocation += relocation
        return total_relocation / max_flows, cum_dst

    def _restoration_dijsktra_weighted(self, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        node_weight = self._restoration_node_weight
        edge_dict = self._restoration_edge_dict
        shortest_paths = {initial: (None, node_weight[initial])}
        current_node = initial
        visited = set()
        while current_node != end:
            visited.add(current_node)
            # destinations = graph.edges[current_node]
            destinations = edge_dict[current_node]
            weight_to_current_node = shortest_paths[current_node][1]
            for next_node in destinations:
                weight = node_weight[next_node] + weight_to_current_node
                # weight = 1 + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)
            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                print(f'error:{initial}---{end}')
                return False
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        # return path
        return shortest_paths[end][1], path

    def flow_weighted_restoration(self, disruption_k, disruption_strategy, simulation_num):
        # disruption strategies include:
        # 'node_degree', 'node_flow', 'node_betweenness_centrality', 'node_flow_centrality'
        # return restoration curve, format: list of tuples [(step num, restoration, restoration + relocation),...]
        def rafr_total_flow_calculator(flow_dict, od_cost, node_list):
            max_flows = 0
            for key, item in flow_dict.items():
                max_flows += item
            total_flows = 0
            for x in node_list:
                for y in node_list:
                    if od_cost[x, y] < 1:  # edges are bi-directional
                        total_flows += flow_dict[x, y]
            return total_flows / max_flows  # normalized flow

        def linear_interpolation(curve):
            pn = 0
            while pn < len(curve) - 1:
                x1, x2 = curve[pn][0], curve[pn + 1][0]
                dx = x2 - x1
                if dx > 1:
                    y1, y2, z1, z2 = curve[pn][1], curve[pn + 1][1], curve[pn][2], curve[pn + 1][2]
                    dy = (y2 - y1) / dx
                    dz = (z2 - z1) / dx
                    for m in range(1, dx):
                        curve.insert(pn + m, (x1 + m, y1 + m * dy, z1 + m * dz))
                    pn += dx
                    # print(f'Warning: linear interpolation applied between point {x1} and {x1 + dx}')
                else:
                    pn += 1
            return curve

        curves = []
        k_nodes = int(self.G.number_of_nodes() * disruption_k)
        for simulation_no in range(simulation_num):
            print(f'Simulation No. {simulation_no + 1} / {simulation_num}')
            v_b = self._attack_sequence_generation(strategy=disruption_strategy, multiple_removal=1)[0:k_nodes]
            v_a = []
            self._restoration_node_weight = {}
            # used as node_weight for path finding; 0 for operational, 1 for disrupted.
            for node in self.get_node_list():
                if node not in v_b:
                    v_a.append(node)
                    self._restoration_node_weight[node] = 0
                else:
                    self._restoration_node_weight[node] = 1
            print(f'{len(v_a)} operational nodes, {len(v_b)} disrupted nodes')
            pbar = tqdm(total=len(v_b), desc=f"Simulation No.{simulation_no + 1}/{simulation_num}")
            self._restoration_edge_dict = self.get_edge_dict()
            od_cost, paths = {}, {}  # initialize OD path recovery cost; represent the number of disrupted nodes on path
            curve = []
            steps = 0  # x value for curve; cumulative od_cost in steps
            cum_dst = None  # cumulative walking distance for relocation
            while v_b:  # recovery cycle
                for x in self.get_node_list():  # od_cost computation
                    for y in self.get_node_list():
                        # initialize or update od_cost and paths
                        if (x, y) not in od_cost.keys():
                            # od pairs are kept which have been connected in previous cycles
                            od_cost[x, y], paths[x, y] = self._restoration_dijsktra_weighted(x, y)
                            if od_cost[x, y] == math.inf:
                                print(f'error: path ({x},{y}) not found; ad_matrix may not be complete')
                            else:
                                od_cost[y, x] = od_cost[x, y]  # if all edges are bi-directional
                                paths[y, x] = paths[x, y]
                # evaluate and save initial/current recovery step (determined by previous loop)
                rl, cum_dst = self._flow_weighted_relocation(v_a, v_b, cum_dst)
                rt = rafr_total_flow_calculator(self.od_flow, od_cost, self.get_node_list())
                curve.append([steps, rt, rt + rl])  # update curve
                # determine the next step based on cost_benefit
                cost_benefit = {}
                for x in self.get_node_list():
                    for y in self.get_node_list():
                        if od_cost[x, y] < 1:  # i.e. path connected, no need for recovery
                            cost_benefit[x, y] = -1
                        else:
                            cost_benefit[x, y] = (self.od_flow[x, y] + self.od_flow[y, x]) / od_cost[x, y]
                # identify prior od pair to connect
                best_od = max(cost_benefit, key=lambda p: cost_benefit[p])
                # print(prior_od, cost_benefit[prior_od], od_cost[prior_od])
                step_length = od_cost[best_od]
                steps += step_length  # update steps; increased value = od_cost
                pbar.update(step_length)
                path = paths[best_od]  # the least cost path
                for node in path:  # update node sets, node state and edge_dict
                    if node in v_b:
                        v_a.append(node)
                        v_b.remove(node)
                        self._restoration_node_weight[node] = 0
                od_cost_prev = od_cost  # keep copy of od_cost
                od_cost = {}  # reset od_cost
                for x in self.get_node_list():  # keep od pairs that has already been connected
                    for y in self.get_node_list():
                        if od_cost_prev[x, y] == 0:
                            od_cost[x, y] = 0
            # evaluate and save the last step
            for x in self.get_node_list():  # od_cost computation for od pairs previously unconnected
                for y in self.get_node_list():
                    # od pairs are kept which have been connected in previous cycles
                    if (x, y) not in od_cost.keys():
                        od_cost[x, y], paths[x, y] = self._restoration_dijsktra_weighted(x, y)
                        if od_cost[x, y] is False:
                            print(f'error: path ({x},{y}) not found; ad_matrix may not be complete')
                        else:
                            od_cost[y, x] = od_cost[x, y]  # if all edges are bi-directional
                            paths[y, x] = paths[x, y]
            rl, cum_dst = self._flow_weighted_relocation(v_a, v_b, cum_dst)
            rt = rafr_total_flow_calculator(self.od_flow, od_cost, self.get_node_list())
            curve.append((steps, rt, rt + rl))  # update curve
            pbar.close()
            curves.append(curve)
        for cn in range(len(curves)):
            curves[cn] = linear_interpolation(curves[cn])
        curves = np.mean(curves, axis=0)
        return curves

    def plot(self, show=False, save=False, save_path=None,
             with_labels=False, caption=False, resize=1.0, legend=False, clean_after_drawing=True,
             alpha=1,
             node_size=3,
             node_color='#1f78b4',
             node_shape='o',
             linewidths=0.2,
             width=0.2,
             edge_color='dimgray',
             style='solid',
             arrowsize=1,
             font_size=6,
             edge_cmap=None,
             edge_vmin=None,
             edge_vmax=None,
             color_by_label=None):
        plt.figure(1, figsize=(12, 12))
        pos = {key: (value[1], value[0]) for key, value in self.node_coordinates.items()}
        if color_by_label:
            edge_color = []
            for (u, v, attrib_dict) in list(self.G.edges.data()):
                edge_color.append(color_by_label[attrib_dict['label']])
            node_color = []
            for u, attrib_dict in list(self.G.nodes.data()):
                node_color.append(color_by_label[attrib_dict['label']])
        nx.draw_networkx(self.G, pos=pos,
                         with_labels=with_labels,
                         alpha=alpha,
                         node_size=node_size,
                         node_color=node_color,
                         node_shape=node_shape,
                         linewidths=linewidths,
                         width=width,
                         edge_color=edge_color,
                         style=style,
                         arrowsize=arrowsize,
                         font_size=font_size,
                         font_family='Times',
                         label=self.name,
                         edge_cmap=edge_cmap,
                         edge_vmin=edge_vmin,
                         edge_vmax=edge_vmax)
        if legend:
            plt.legend()
        if caption:
            plt.title(self.name)
        if save_path is not None:
            plt.savefig(save_path, transparent=True, dpi=300)
        elif save:
            plt.savefig(f'fig_{self.name}.png', transparent=True, dpi=300 * resize)
        if show:
            plt.show()
        if clean_after_drawing:
            plt.close()

    def robustness_based_on_routes_data(self, removal='edge', strategy='random', weight=None,
                                        number_of_tests=100,
                                        multiple_removal=1,
                                        multi_processing=False,
                                        export_raw_results_to=None,
                                        sustaining_level='weak'):
        """
        :param removal: "node" or "edge"
        :param strategy: "random"
        :param weight:
        :param number_of_tests: # of repeated tests
        :param multiple_removal: # of nodes/edges to be removed in each step
        :param multi_processing: N/A
        :param export_raw_results_to: export raw results of all repeated tests
        :param sustaining_level: "strong" or "weak"
        :return:
        """
        ys, xs = [], []
        for test in range(number_of_tests):
            temp = copy.deepcopy(self)
            y, x = [1.0], [0.0]
            y_max = temp.network.compute_total_capacity()
            if removal == 'edge' and strategy == 'random':
                edges = copy.deepcopy(temp.network.edge_set())
                total_steps = len(edges)
                step = 0
                pbar = tqdm(total=total_steps)
                while edges:
                    if multiple_removal < len(edges):
                        i = random.sample(list(edges), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(edges))
                    temp.network.remove_edges_from(i, level=sustaining_level)
                    edges = edges - set(i)
                    y.append(temp.network.compute_total_capacity() / y_max)
                    step += len(i)
                    x.append(step / total_steps)
                    pbar.update(len(i))
                pbar.close()
            elif removal == 'node' and strategy == 'random':
                stops = copy.deepcopy(temp.network.stop_set())
                total_steps = len(stops)
                step = 0
                pbar = tqdm(total=total_steps)
                while stops:
                    if multiple_removal < len(stops):
                        i = random.sample(list(stops), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(stops))
                    temp.network.remove_stops_from(i, level=sustaining_level)
                    stops = stops - set(i)
                    y.append(temp.network.compute_total_capacity() / y_max)
                    step += len(i)
                    x.append(step / total_steps)
                    pbar.update(len(i))
                pbar.close()
            else:
                pass
            ys.append(y)
            xs.append(x)
        if export_raw_results_to:
            export_list(ys, filename=export_raw_results_to)
        results = (np.mean(ys, axis=0), np.mean(xs, axis=0), np.std(ys, axis=0))
        print('rb=', numerical_integral_nml(results[0], results[1]))
        print(results[0])
        print(results[1])
        print(results[2])
        return results

    def relocation_integrated_robustness_based_on_routes_data(self, removal='edge', strategy='random', weight=None,
                                                              number_of_tests=100,
                                                              multiple_removal=1,
                                                              multi_processing=False,
                                                              export_raw_results_to=None,
                                                              sustained_level='weak'):
        """
        :param removal: "node" or "edge"
        :param strategy: "random"
        :param weight:
        :param number_of_tests: # of repeated tests
        :param multiple_removal: # of nodes/edges to be removed in each step
        :param multi_processing: N/A
        :param export_raw_results_to: export raw results of all repeated tests
        :param sustaining_level: "strong" or "weak"
        :return:
        """

        def _relocation_capacity(rr_dict):
            relocation = {}
            for relocated_node, rvs in rr_dict.items():
                nb = [r_v[0] for r_v in rvs]
                dst = [float(rv[1]) for rv in rvs]
                rr = [float(rv[2]) for rv in rvs]
                for k, received_node in enumerate(nb):
                    if received_node not in relocation.keys():
                        relocation[received_node] = rr[k] * original_node_cap[relocated_node]
                    else:
                        relocation[received_node] += rr[k] * original_node_cap[relocated_node]
            total_relocation = np.sum([v for v in relocation.values()])
            return total_relocation / y_max

        original_node_cap = self.network.compute_node_capacity()
        original_stop_set = self.network.stop_set()
        path_to_neighbor_dict = 'mptn_analyze_relocation_results/neighbor'
        self.network.generate_neighborhood_dict(path_to_save=path_to_neighbor_dict)
        logs = [['Disruption level', 'Remaining capacity', 'Relocation',
                 'Number of fractured trips', 'Length of fractured trips',
                 'Average degree', 'Efficiency', 'No_test', 'Parameter']]
        ys, xs = [], []
        for test in range(number_of_tests):
            temp = copy.deepcopy(self)
            logs.append([0.0, 1.0, 0.0, 0.0, 0.0,
                         temp.G.number_of_edges() / temp.G.number_of_nodes(),
                         nx.global_efficiency(temp.G.to_undirected()),
                         test, f'{removal.capitalize()} - {sustained_level.capitalize()}'])
            print(logs[1])
            y, x = [1.0], [0.0]
            y_max = temp.network.compute_total_capacity()

            if removal == 'edge' and strategy == 'random':
                edges = copy.deepcopy(temp.network.edge_set())
                total_steps = len(edges)
                step = 0
                pbar = tqdm(total=total_steps)
                while edges:
                    if multiple_removal < len(edges):
                        i = random.sample(list(edges), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(edges))
                    fractured_trips = temp.network.remove_edges_from(i, level=sustained_level)
                    temp.update_graph_by_routes_data()
                    fractured_length = [len(trip) for trip in fractured_trips]
                    number_of_fractured_routes = np.sum(fractured_length)
                    fractured_route_length = [len(fracture) for trip in fractured_trips for fracture in trip if
                                              len(fracture) > 0]
                    # print(fractured_route_length)
                    # for trip in fractured_trips:
                    #     print(trip)
                    if fractured_route_length:
                        length_of_fractured_routes = np.mean(fractured_route_length)
                    else:
                        length_of_fractured_routes = 0.0
                    y_k = temp.network.compute_total_capacity() / y_max
                    step += len(i)
                    x_k = step / total_steps
                    if temp.G.number_of_nodes() > 0:
                        average_degree = temp.G.number_of_edges() / temp.G.number_of_nodes()
                    else:
                        average_degree = 0
                    e_n, e_on = temp.G.number_of_nodes(), self.G.number_of_nodes()
                    e = nx.global_efficiency(temp.G.to_undirected()) * (e_n * (e_n - 1)) / (e_on * (e_on - 1))
                    disrupted_nodes = list(original_stop_set - temp.network.stop_set())
                    rr_dict = temp.network.compute_relocation_rate(disrupted_node_list=disrupted_nodes, max_dst=750,
                                                                   read_previous_neighbor_dict=path_to_neighbor_dict,
                                                                   single_disruption_analysis=False)
                    relocation_capacity = _relocation_capacity(rr_dict)
                    log = [x_k, y_k, relocation_capacity, number_of_fractured_routes, length_of_fractured_routes,
                           average_degree, e, test, f'{removal.capitalize()} - {sustained_level.capitalize()}']
                    print(log)
                    logs.append(log)

                    edges = edges - set(i)
                    y.append(y_k)
                    x.append(x_k)
                    pbar.update(len(i))
                pbar.close()

            elif removal == 'node' and strategy == 'random':
                stops = copy.deepcopy(temp.network.stop_set())
                total_steps = len(stops)
                step = 0
                pbar = tqdm(total=total_steps)
                while stops:
                    if multiple_removal < len(stops):
                        i = random.sample(list(stops), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(stops))
                    fractured_trips = temp.network.remove_stops_from(i, level=sustained_level)
                    temp.update_graph_by_routes_data()
                    fractured_length = [len(trip) for trip in fractured_trips]
                    number_of_fractured_routes = np.sum(fractured_length)
                    fractured_route_length = [len(fracture) for trip in fractured_trips for fracture in trip if
                                              len(fracture) > 0]
                    if fractured_route_length:
                        length_of_fractured_routes = np.mean(fractured_route_length)
                    else:
                        length_of_fractured_routes = 0.0
                    y_k = temp.network.compute_total_capacity() / y_max
                    step += len(i)
                    x_k = step / total_steps
                    if temp.G.number_of_nodes() > 0:
                        average_degree = temp.G.number_of_edges() / temp.G.number_of_nodes()
                    else:
                        average_degree = 0
                    e_n, e_on = temp.G.number_of_nodes(), self.G.number_of_nodes()
                    e = nx.global_efficiency(temp.G.to_undirected()) * (e_n * (e_n - 1)) / (e_on * (e_on - 1))
                    disrupted_nodes = list(original_stop_set - temp.network.stop_set())
                    rr_dict = temp.network.compute_relocation_rate(disrupted_node_list=disrupted_nodes, max_dst=750,
                                                                   read_previous_neighbor_dict=path_to_neighbor_dict,
                                                                   single_disruption_analysis=False)
                    relocation_capacity = _relocation_capacity(rr_dict)
                    log = [x_k, y_k, relocation_capacity, number_of_fractured_routes, length_of_fractured_routes,
                           average_degree, e, test, f'{removal.capitalize()} - {sustained_level.capitalize()}']
                    print(log)
                    logs.append(log)
                    stops = stops - set(i)
                    y.append(y_k)
                    x.append(x_k)
                    pbar.update(len(i))
                pbar.close()
            else:
                raise (KeyError, 'removal = edge or node, and strategy = random')
            ys.append(y)
            xs.append(x)
        if export_raw_results_to:
            export_list(ys, filename=export_raw_results_to)
        results = (np.mean(ys, axis=0), np.mean(xs, axis=0), np.std(ys, axis=0))
        print('rb=', numerical_integral_nml(results[0], results[1]))
        print(results[0])
        print(results[1])
        print(results[2])
        return results, logs

    def single_removal(self, removal='edge', sustained_level='strong'):

        logs = [['removed_node_id', 'Remaining_Capacity',
                 'number_of_fractions', 'length_of_fractured_route',
                 'edge_density', 'efficiency']]
        y_max = self.network.compute_total_capacity()
        if removal == 'edge':
            edges = copy.deepcopy(list(self.network.edge_set()))
            for i in tqdm(edges):
                temp = copy.deepcopy(self)
                fractured_trips = temp.network.remove_edges_from([i], level=sustained_level)
                temp.update_graph_by_routes_data()
                fractured_length = [len(trip) for trip in fractured_trips]
                number_of_fractions = sum(fractured_length) - len(fractured_length)
                length_of_fractured_route = [len(fracture) for trip in fractured_trips for fracture in trip]
                y = temp.network.compute_total_capacity() / y_max
                edge_density = temp.G.number_of_edges() / temp.G.number_of_nodes()
                e = nx.global_efficiency(temp.G.to_undirected())
                log = [i, y, number_of_fractions, length_of_fractured_route,
                       edge_density, e]
                print(log)
                logs.append(log)
        elif removal == 'node':
            stops = copy.deepcopy(list(self.network.stop_set()))
            for i in tqdm(stops):
                temp = copy.deepcopy(self)
                fractured_trips = temp.network.remove_stops_from([i], level=sustained_level)
                temp.update_graph_by_routes_data()
                fractured_length = [len(trip) for trip in fractured_trips]
                number_of_fractions = sum(fractured_length) - len(fractured_length)
                length_of_fractured_route = [len(fracture) for trip in fractured_trips for fracture in trip]
                y = temp.network.compute_total_capacity() / y_max
                edge_density = temp.G.number_of_edges() / temp.G.number_of_nodes()
                e = nx.global_efficiency(temp.G.to_undirected())
                log = [i, y, number_of_fractions, length_of_fractured_route,
                       edge_density, e]
                print(log)
                logs.append(log)
        return logs

    def relocation_based_on_routes_data_outdated(self, removal='edge', strategy='random', weight=None,
                                                 number_of_tests=100,
                                                 multiple_removal=1,
                                                 multi_processing=False,
                                                 export_raw_results_to_folder=None,
                                                 sustaining_level='weak',
                                                 path_to_neighbor_dict='mptn_analyze_relocation_results/neighbor'):
        """
        old algorithm
        :param export_raw_results_to_folder:
        :param path_to_neighbor_dict:
        :param removal: "node" or "edge"
        :param strategy: "random"
        :param weight:
        :param number_of_tests: # of repeated tests
        :param multiple_removal: # of nodes/edges to be removed in each step
        :param multi_processing: N/A
        :param export_raw_results_to: export raw results of all repeated tests
        :param sustaining_level: "strong" or "weak"
        :return:
        """
        ys, xs, rs = [], [], []
        for test in range(number_of_tests):
            temp = copy.deepcopy(self)
            y, x, r = [1.0], [0.0], [0.0]
            y_max = temp.network.compute_total_capacity()
            if removal == 'edge':
                edges = copy.deepcopy(temp.network.edge_set())
                total_steps = len(edges)
                step = 0
                pbar = tqdm(total=total_steps)
                while edges:
                    if multiple_removal < len(edges):
                        i = random.sample(list(edges), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(edges))
                    temp.network.remove_edges_from(i, level=sustaining_level)
                    edges = edges - set(i)
                    y.append(temp.network.compute_total_capacity() / y_max)
                    step += len(i)
                    x.append(step / total_steps)
                    pbar.update(len(i))
                pbar.close()
            elif removal == 'node':

                stops = copy.deepcopy(temp.network.stop_set())
                total_steps = len(stops)
                step = 0
                pbar = tqdm(total=total_steps)
                ###################
                removed = []
                original_node_cap = self.network.compute_node_capacity()

                ###################
                while stops:
                    if strategy == 'node_capacity':
                        nc_dic = temp.network.compute_node_capacity()
                        i = search_for_max(nc_dic, multiple_search=multiple_removal)
                    elif multiple_removal < len(stops):
                        i = random.sample(list(stops), k=multiple_removal)
                    else:
                        i = copy.deepcopy(list(stops))
                    temp.network.remove_stops_from(i, level=sustaining_level)
                    ###################
                    relocation = {}
                    removed.extend(i)
                    rr_dict = temp.network.compute_relocation_rate(disrupted_node_list=i, max_dst=750,
                                                                   path_to_save_neighbor_dict=path_to_neighbor_dict,
                                                                   read_previous_neighbor_dict=path_to_neighbor_dict,
                                                                   single_disruption_analysis=False)
                    for relocated_node, rvs in rr_dict.items():
                        nb = [r_v[0] for r_v in rvs]
                        dst = [float(rv[1]) for rv in rvs]
                        rr = [float(rv[2]) for rv in rvs]
                        for k, received_node in enumerate(nb):
                            if received_node not in relocation.keys():
                                relocation[received_node] = rr[k] * original_node_cap[relocated_node]
                            else:
                                relocation[received_node] += rr[k] * original_node_cap[relocated_node]
                    total_relocation = np.sum([v for v in relocation.values()])
                    r.append(total_relocation / y_max)
                    # print(r)
                    ###################
                    stops = stops - set(i)
                    y.append(temp.network.compute_total_capacity() / y_max)
                    step += len(i)
                    x.append(step / total_steps)
                    pbar.update(len(i))
                pbar.close()
            else:
                raise ValueError(f'{removal}, removal="node" or "edge"')
            ys.append(y)
            xs.append(x)
            rs.append(r)
        if export_raw_results_to_folder:
            export_list(ys, filename=export_raw_results_to_folder + 'raw_capacity_curve.csv')
            export_list(rs, filename=export_raw_results_to_folder + 'raw_relocation_curve.csv')
        results = (
            np.mean(ys, axis=0), np.mean(xs, axis=0), np.std(ys, axis=0), np.mean(rs, axis=0), np.std(rs, axis=0))
        print('rb=', numerical_integral_nml(results[0], results[1]))
        print(results[0])
        print(results[1])
        print(results[2])
        print(results[3])
        print(results[4])
        return results


def create_ER_benchmark_for_graph(G, seed=616, undirected_benchmark=False,
                                  average_edge_length=None, std_edge_length=None,
                                  dict_of_node_lat_lon=None,
                                  samples=None):
    # Erdos-Renyi Gnm model, modified based on networkx.gnm_random_graph()
    import random

    def searchInsert(nums, target):
        n = len(nums)
        if target > nums[-1]:
            return -1
        elif target < nums[0]:
            return 0
        left, right = 0, n - 1
        ans = -2
        while left <= right:
            mid = (left + right) // 2
            if target <= nums[mid]:
                right = mid - 1
                ans = mid
            else:
                left = mid + 1
        if ans == -2:
            raise KeyError
        return ans

    def discrete_distribution(samples, bin_width):
        samples = list(samples)
        vmax = max(samples)
        vmin = min(samples)
        bins = [c for c in np.arange(vmin, vmax + bin_width, bin_width)]
        freq = [0 for c in np.arange(vmin, vmax + bin_width, bin_width)]
        for c in samples:
            bin = round((c - vmin) / bin_width)
            freq[int(bin)] += 1
        tot = sum(freq)
        prob = [f / tot for f in freq]
        return bins, prob

    if average_edge_length and std_edge_length and dict_of_node_lat_lon:
        # geospatial constraint by normal distribution
        geospatial_constraint = True
    else:
        geospatial_constraint = False
    temp_dst = {n: {} for n in G.nodes}  # temporary dict for lengths of potential edges to avoid repeated computation

    random.seed(seed)
    nodes = list(G.nodes())
    n, m = G.number_of_nodes(), G.number_of_edges()
    if G.is_directed():
        if undirected_benchmark:
            BG = nx.Graph()
            m /= 2.0
        else:
            BG = nx.DiGraph()
    else:
        # undirected
        BG = nx.Graph()
    BG.add_nodes_from(nodes)
    if n == 1:
        return BG
    max_edges = n * (n - 1)
    if not G.is_directed() or undirected_benchmark:
        max_edges /= 2.0
    if m >= max_edges:
        # regardless of geospatial constraint
        return nx.complete_graph(n, create_using=BG)
    nlist = list(BG)
    edge_count = 0
    elist = list(combinations(nlist, 2))
    elist_dst = {(u, v): haversine(*(dict_of_node_lat_lon[u] + dict_of_node_lat_lon[v])) for u, v in elist}
    sorted_elist_dst = sorted(elist_dst.items(), key=lambda obj: obj[1])
    sorted_elist = [item[0] for item in sorted_elist_dst]
    sorted_dst = [item[1] for item in sorted_elist_dst]
    population, weights = discrete_distribution(samples=samples, bin_width=100)
    samples = random.choices(population, weights, k=int(m+1))
    while edge_count < m:
        # ran = random.gauss(mu=average_edge_length, sigma=std_edge_length)
        ran = samples[edge_count]
        pos = searchInsert(sorted_dst, ran)
        if pos > 0 and sorted_dst[pos] - ran > ran - sorted_dst[pos-1]:
            pos = pos - 1
        edge, dst = sorted_elist[pos], sorted_dst[pos]
        BG.add_edge(*edge)
        edge_count = edge_count + 1
        del sorted_elist[pos]
        del sorted_dst[pos]

    '''while edge_count < m:
        # generate random edge,u,v
        u = random.choice(nlist)
        v = random.choice(nlist)
        if u == v or BG.has_edge(u, v):
            continue
        # u, v = elist.pop(random.randint(0, len(elist)-1))
        elif geospatial_constraint:
            if v in temp_dst[u].keys():
                dst = temp_dst[u][v]
            else:
                coord = dict_of_node_lat_lon[u] + dict_of_node_lat_lon[v]
                dst = haversine(*coord)
                temp_dst[u][v] = dst
                temp_dst[v][u] = dst
            z = (dst - average_edge_length) / std_edge_length
            p = scipy.stats.norm.sf(abs(z)) * 2
            rp = random.random()
            if rp > p:
                # elist.append((u, v))
                continue
            else:
                BG.add_edge(u, v)
                edge_count = edge_count + 1
        else:
            BG.add_edge(u, v)
            edge_count = edge_count + 1'''
    return BG


def global_efficiency_modified_for_unconnected_digragh(G):
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G)
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        g_eff = 0
    return g_eff


def average_shortest_path_length_modified_for_unconnected_digragh(G):
    n = len(G)
    denom = n * (n - 1)
    # for u in nodes:
    #   for v in nodes:
    #
    # for u, v in combinations(list(G.nodes), 2):
    #
    # lmax = maximal_shortest_path_length(G)
    if denom != 0:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        lmax = max([distance for source, targets in lengths.items() for target, distance in targets.items()])
        nodes = list(G.nodes)
        cpl = 0
        for u in nodes:
            for v in nodes:
                if v in lengths[u].keys():
                    dst = lengths[u][v]
                    if dst > 0:
                        cpl += dst
                else:
                    cpl += lmax
        cpl /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        cpl = 0
    return cpl
def maximal_shortest_path_length(G):
    lengths = nx.all_pairs_shortest_path_length(G)
    return max([distance for source, targets in lengths for target, distance in targets.items()])

# def average_shortest_path_length_modified_for_unconnected_digragh(G):
#     def path_length(v):
#         return nx.single_source_shortest_path_length(G, v)
#     n = len(G)
#     denom = n * (n - 1)
#     s = sum(l for u in G for l in path_length(u).values() if l > 0)
#     s /= denom
#     return s

def search_for_max(dic, multiple_search=1):
    # list(dict(sorted(dic.items(), key=lambda item: item[1])).keys())[-1]
    result = []
    temp_dic = copy.deepcopy(dic)
    while temp_dic and len(result) < multiple_search:
        max_value = max(temp_dic.values())
        max_value_keys = []
        for key, value in temp_dic.items():
            if value == max_value:
                max_value_keys.append(key)
        the_key = random.choice(max_value_keys)
        result.append(the_key)
        temp_dic.pop(the_key)
    return result


def get_edge_dict_from(G):
    edge_dict = defaultdict(list)
    for edge in G.edges():
        x, y = edge[0], edge[1]
        edge_dict[x].append(y)
    return edge_dict


def numerical_integral_nml(curve, xs=None, dec=None):
    """
    Riemann sum Midpoint rule
    :param curve:
    :param xs:
    :param dec:
    :return:
    """
    y = np.asarray(curve)
    if xs is not None:
        if len(curve) != len(xs):
            print('error: curve and xs have different length')
            return
        else:
            x = np.asarray(xs)
            nml = 0.5 * ((x[1] - x[0]) * y[0] + np.sum((x[2:] - x[:-2]) * y[1:-1]) + (x[-1] - x[-2]) * y[-1])
    else:
        n = len(y) - 1
        nml = (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1])) / n
    if dec:
        return round(nml, dec)
    else:
        return nml


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = np.asarray(array).astype(float).flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def dijsktra_weighted(edge_dict, edge_weight, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()
    while current_node != end:
        visited.add(current_node)
        # destinations = graph.edges[current_node]
        destinations = edge_dict[current_node]
        weight_to_current_node = shortest_paths[current_node][1]
        for next_node in destinations:
            weight = edge_weight[(current_node, next_node)] + weight_to_current_node
            # weight = 1 + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)
        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return math.inf
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])
    # Work back through destinations in shortest path
    # path = []
    # while current_node is not None:
    #     path.append(current_node)
    #     next_node = shortest_paths[current_node][0]
    #     current_node = next_node
    # Reverse path
    # path = path[::-1]
    # return path
    return shortest_paths[end][1]


def combine_network_weighted(XC1, XC2, cross_layer_edges, new_name='Combined_network'):
    # not support flow
    CN = Resilience(new_name)
    CN.G.add_edges_from(list(XC1.G.edges()))
    CN.G.add_edges_from(list(XC2.G.edges()))
    CN.G.add_edges_from(cross_layer_edges)
    CN.node_coordinates.update(XC1.node_coordinates)
    CN.node_coordinates.update(XC2.node_coordinates)
    print(f'{CN.name}, '
          f'number of nodes = {CN.G.number_of_nodes()}, '
          f'number of edges = {CN.G.number_of_edges()}')
    return CN


def line_plot(xs, ys, labels=None, axis_labels=None, save_path=None):
    if np.asarray(xs).ndim == 1:
        xs = [xs]
    if np.asarray(ys).ndim == 1:
        ys = [ys]
        number_of_lines = 1
    else:
        number_of_lines = len(ys)
    if labels is None:
        labels = [None] * number_of_lines
    if axis_labels is None:
        axis_labels = ['x', 'y']
    figure = plt.figure(figsize=(6, 4))
    ax0 = figure.add_subplot(1, 1, 1)
    markers = ['o'] + ['^'] + ['s'] + ['x'] + ['+'] + ['h']
    colors = ['k'] + ['b'] + ['darkgreen'] + ['r'] + ['darkorange'] + ['m'] + ['c'] + ['y']
    linestyles = ['-'] + ['--'] + [':'] + ['-.']
    for i in range(number_of_lines):
        ax0.plot(xs[i], ys[i], label=labels[i], color='grey',
                 marker=markers[i],
                 alpha=0.8, mfc='none', mec=colors[i],
                 markersize=4, linewidth=0.5)
    ax0.legend(ncol=2, prop={'size': 10.5})
    ax0.set_xlabel(axis_labels[0])
    ax0.set_ylabel(axis_labels[1])
    plt.subplots_adjust(left=0.12, bottom=0.16, right=0.96, top=0.92, wspace=0.33, hspace=0.4)
    if save_path:
        plt.savefig(save_path, transparent=True, dpi=300)
    else:
        plt.show()


def plot_distribution(list_of_value):
    fig = plt.figure(figsize=(9, 3))
    plt.hist(list_of_value, bins=20)
    plt.show()

def harmonic_mean(array):
    return len(array) / sum([1/v for v in array])
