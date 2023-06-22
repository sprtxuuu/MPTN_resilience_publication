import copy

import xc_resilience_live as xc
from mptn_modeling_from_gtfs import create_mptn_model
from results_plotting_tool import plot_centrality_distribution, plot_degree_distribution


def geospatial_efficiency(G, dict_of_node_lat_lon):
    def haversine(lat1, lon1, lat2, lon2):  # (decimal）
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        dst = c * r * 1000
        return dst

    import networkx as nx
    from copy import deepcopy
    geo_eff = 0
    dG = deepcopy(G)
    n = len(dG)
    denom = n * (n - 1)
    if denom != 0:
        for u, v in dG.edges:
            # # # coord=(lat1, lon1, lat2, lon2)
            coord = dict_of_node_lat_lon[u] + dict_of_node_lat_lon[v]
            dG[u][v]['d'] = haversine(*coord)
        lengths = dict(nx.all_pairs_dijkstra_path_length(dG, weight='d'))
        for source, targets in lengths.items():
            for target, distance in targets.items():
                if distance > 0:
                    coord = dict_of_node_lat_lon[source] + dict_of_node_lat_lon[target]
                    # # # flight distance / travel distance
                    geo_eff += haversine(*coord) / distance
        geo_eff /= denom
    else:
        geo_eff = 0
    return geo_eff


def edge_length_distribution(G, dict_of_node_lat_lon):
    def haversine(lat1, lon1, lat2, lon2):  # (decimal）
        from math import radians, cos, sin, asin, sqrt
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        dst = c * r * 1000
        return dst

    import networkx as nx
    from numpy import mean, std
    from copy import deepcopy
    geo_eff = 0
    dG = deepcopy(G)
    denom = dG.number_of_edges()
    dsts = []
    if denom != 0:
        for u, v in dG.edges():
            coord = dict_of_node_lat_lon[u] + dict_of_node_lat_lon[v]
            dsts.append(haversine(*coord))
    else:
        return 0, 0
    return mean(dsts), std(dsts), dsts


def benchmark_expanding_unweighted_network(analyze_topology_and_GINI=1,
                                           analyze_node_metric_distribution=0,
                                           analyze_rb=0,
                                           imt_edges=0):
    """
    run analysis
    :param analyze_topology_and_GINI: True/False
    :param analyze_node_metric_distribution: True/False
    :param analyze_rb: True/False
    :param imt_edges: True/False, enable intermodal edges (interconnected or isolated MPTN)
    :return: N/A
    """

    def check_stop_label(print_items=False):
        i = 0
        for stop in mptn.network.stop_repository.values():
            if stop.label is None:
                stop.show()
            if print_items:
                print(stop.label)

    all_modes = [['MTR'],
                 ['CTB', 'KMB+CTB', 'KMB', 'NLB', 'PI', 'LWB', 'DB', 'LRTFeeder', 'KMB+NWFB', 'LWB+CTB', 'NWFB', 'XB'],
                 ['GMB'],
                 ['LR'],
                 ['FERRY'],
                 ['TRAM']]

    names = ['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM']
    log = []
    for i in [0, 1, 2, 3, 4, 5]:
        mode = all_modes[:i + 1]
        print(mode)
        modes_to_remove = [item for item in all_modes if item not in mode]
        print(modes_to_remove)
        # import mptn network model
        mptn = create_mptn_model()
        check_stop_label()
        for item in modes_to_remove:
            for subitem in item:
                mptn.network.remove_routes_by_agency(subitem)

        if imt_edges:
            intermodal_edges = mptn.network.generate_intermodal_edges(dst_limit=100)
            mptn.update_graph_by_routes_data(intermodal_edge_list=intermodal_edges)
        else:
            mptn.update_graph_by_routes_data()
        mptn.network.show()

        index = (mptn.G.to_undirected().number_of_edges() * 2 - mptn.G.number_of_edges()) / mptn.G.number_of_edges()
        print('index=', index)

        if analyze_topology_and_GINI:
            node_coordinates = {id: stop.coordinates() for id, stop in mptn.network.stop_repository.items()}
            # preps = [['Label', '|V|', '|E|', 'GINI_nd', 'GINI_bc', 'E']]
            preps = [['Label', '|V|', '|E|', 'GINI_nd', 'GE', 'E', 'AEL', 'STDEL']]
            ael, stdel, samples = edge_length_distribution(mptn.G, node_coordinates)
            preps.append(['System',
                          mptn.G.number_of_nodes(),
                          mptn.G.number_of_edges(),
                          round(mptn.preparedness_node_degree_gini(), 4),
                          round(mptn.preparedness_node_betweenness_centrality_gini(), 4),
                          geospatial_efficiency(mptn.G, node_coordinates),
                          xc.global_efficiency_modified_for_unconnected_digragh(mptn.G),
                          ael, stdel])
            for nb in range(100):
                BG = xc.create_ER_benchmark_for_graph(G=mptn.G, seed=nb, undirected_benchmark=True,
                                                      dict_of_node_lat_lon=node_coordinates,
                                                      samples=samples
                                                      ).to_directed()
                mptn.G = BG
                ael_t, stdel_t, samples_t = edge_length_distribution(mptn.G, node_coordinates)
                preps.append(['Benchmark',
                              mptn.G.number_of_nodes(),
                              mptn.G.number_of_edges(),
                              round(mptn.preparedness_node_degree_gini(), 4),
                              round(mptn.preparedness_node_betweenness_centrality_gini(), 4),
                              geospatial_efficiency(mptn.G, node_coordinates),
                              xc.global_efficiency_modified_for_unconnected_digragh(mptn.G),
                              ael_t, stdel_t])
            for line in preps:
                print(line)
            # xc.export_list(xp_list=preps, filename=f'Benchmark_ER_results/expanding_imt_{imt_edges}_step_{i}.csv')
            xc.export_list(xp_list=preps,
                           filename=f'Benchmark_gs_ER_results/expanding_imt_{imt_edges}_step_{i}_geospatial_efficiency.csv')

        if analyze_node_metric_distribution:
            plot_centrality_distribution(mptn,
                                         save_figure_as=f'Benchmark_gs_ER_results/fig_centrality_distribution_{i}.png', )
            plot_degree_distribution(mptn,
                                     save_figure_as=f'Benchmark_gs_ER_results/fig_degree_distribution_{i}.png')

        if analyze_rb:
            print(mptn.G.number_of_nodes(), mptn.G.number_of_edges())
            node_coordinates = {id: stop.coordinates() for id, stop in mptn.network.stop_repository.items()}
            ael, stdel, samples = edge_length_distribution(mptn.G, node_coordinates)
            for nb in range(100):
                BG = xc.create_ER_benchmark_for_graph(G=mptn.G, seed=nb, undirected_benchmark=True,
                                                      average_edge_length=ael, std_edge_length=stdel,
                                                      dict_of_node_lat_lon=node_coordinates,
                                                      samples=samples).to_directed()
                mptn.G = BG
                print(mptn.G.number_of_nodes(), mptn.G.number_of_edges())
                rb_n_steps = 20
                rb_step_size = round(mptn.G.number_of_nodes() / rb_n_steps)
                if imt_edges:
                    xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=100,
                                                                            multiple_removal=rb_step_size,
                                                                            multi_processing=True),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{i}_rb_rnd_{nb}.csv')
                    xc.export_list(mptn.robustness_unweighted_degree_based_attack(multiple_removal=rb_step_size,
                                                                                  multi_processing=True,
                                                                                  number_of_tests=100),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{i}_rb_nd_{nb}.csv')
                    xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(multiple_removal=rb_step_size,
                                                                                       multi_processing=True,
                                                                                       number_of_tests=10),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{i}_rb_bc_{nb}.csv')
                else:
                    xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=100,
                                                                            multiple_removal=rb_step_size,
                                                                            multi_processing=True),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{i}_rb_rnd_{nb}.csv')
                    xc.export_list(mptn.robustness_unweighted_degree_based_attack(multiple_removal=rb_step_size,
                                                                                  multi_processing=True,
                                                                                  number_of_tests=100),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{i}_rb_nd_{nb}.csv')
                    xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(multiple_removal=rb_step_size,
                                                                                       multi_processing=True,
                                                                                       number_of_tests=10),
                                   filename=f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{i}_rb_bc_{nb}.csv')
            print('\n')


if __name__ == "__main__":
    benchmark_expanding_unweighted_network(analyze_topology_and_GINI=1,
                                           analyze_node_metric_distribution=1,
                                           analyze_rb=1,
                                           imt_edges=0)

    benchmark_expanding_unweighted_network(analyze_topology_and_GINI=1,
                                           analyze_node_metric_distribution=1,
                                           analyze_rb=1,
                                           imt_edges=1)
