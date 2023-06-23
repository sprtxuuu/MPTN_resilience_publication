import xc_resilience_live as xc
from mptn_modeling_from_gtfs import create_mptn_model
from results_plotting_tool import plot_centrality_distribution, plot_degree_distribution


def benchmark_individual_unweighted_network(analyze_topology_and_GINI=0,
                                            analyze_node_metric_distribution=0,
                                            analyze_rb=0):
    """
    run analysis
    :param analyze_topology_and_GINI: True/False
    :param analyze_node_metric_distribution: True/False
    :param analyze_rb: True/False
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
    for i in [0, 1, 2, 3, 4, 5]:
        # import network model
        mode = all_modes[i]
        modes_to_remove = [item for item in all_modes if item != mode]
        mptn = create_mptn_model()
        check_stop_label()
        for item in modes_to_remove:
            for subitem in item:
                mptn.network.remove_routes_by_agency(subitem)
        mptn.update_graph_by_routes_data()
        from benchmark_ER_expanding_unweighted_network import geospatial_efficiency, edge_length_distribution
        if analyze_topology_and_GINI:
            node_coordinates = {id: stop.coordinates() for id, stop in mptn.network.stop_repository.items()}
            preps = [['Label', '|V|', '|E|', 'GINI_nd', 'GINI_bc', 'GE', 'E', 'AEL', 'STDEL']]
            # preps = [['Label', '|V|', '|E|', 'GINI_nd', 'GE', 'E', 'AEL', 'STDEL']]
            ael, stdel, samples = edge_length_distribution(mptn.G, node_coordinates)
            preps.append(['System',
                          mptn.G.number_of_nodes(),
                          mptn.G.number_of_edges(),
                          round(mptn.preparedness_node_degree_gini(), 4),
                          round(mptn.preparedness_node_betweenness_centrality_gini(), 4),
                          geospatial_efficiency(mptn.G, node_coordinates),
                          xc.global_efficiency_modified_for_unconnected_digragh(mptn.G),
                          ael, stdel])
            for line in preps:
                print(line)
            for nb in range(50):
                print(nb)
                BG = xc.create_ER_benchmark_for_graph(G=mptn.G, seed=nb, undirected_benchmark=True,
                                                      average_edge_length=ael, std_edge_length=stdel,
                                                      dict_of_node_lat_lon=node_coordinates,
                                                      samples=samples).to_directed()
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
                print(preps[-1])
            xc.export_list(xp_list=preps, filename=f'Benchmark_gs_ER_results/{names[i]}_geospatial_efficiency.csv')

        if analyze_node_metric_distribution:
            plot_centrality_distribution(mptn,
                                         save_figure_as=f'Benchmark_gs_ER_results/fig_centrality_distribution_{names[i]}.png', )
            plot_degree_distribution(mptn,
                                     save_figure_as=f'Benchmark_gs_ER_results/fig_degree_distribution_{names[i]}.png')

        if analyze_rb:
            for nb in range(100):
                node_coordinates = {id: stop.coordinates() for id, stop in mptn.network.stop_repository.items()}
                ael, stdel, samples = edge_length_distribution(mptn.G, node_coordinates)
                BG = xc.create_ER_benchmark_for_graph(G=mptn.G, seed=nb, undirected_benchmark=True,
                                                      average_edge_length=ael, std_edge_length=stdel,
                                                      dict_of_node_lat_lon=node_coordinates,
                                                      samples=samples).to_directed()
                mptn.G = BG
                rb_n_steps = 40
                rb_step_size = round(mptn.G.number_of_nodes() / rb_n_steps)
                xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=1000,
                                                                        multiple_removal=rb_step_size,
                                                                        multi_processing=True),
                               filename=f'Benchmark_gs_ER_results/robustness/mptn_{names[i]}_rb_rnd_{nb}.csv')
                xc.export_list(mptn.robustness_unweighted_degree_based_attack(multiple_removal=rb_step_size,
                                                                              multi_processing=True,
                                                                              number_of_tests=100),
                               filename=f'Benchmark_gs_ER_results/robustness/mptn_{names[i]}_rb_nd_{nb}.csv')
                xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(multiple_removal=rb_step_size,
                                                                                   multi_processing=True,
                                                                                   number_of_tests=100),
                               filename=f'Benchmark_gs_ER_results/robustness/mptn_{names[i]}_rb_bc_{nb}.csv')


if __name__ == "__main__":
    benchmark_individual_unweighted_network(analyze_topology_and_GINI=1,
                                            analyze_node_metric_distribution=0,
                                            analyze_rb=0)
