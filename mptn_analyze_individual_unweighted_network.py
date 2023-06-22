import xc_resilience_live as xc
from mptn_modeling_from_gtfs import create_mptn_model
from results_plotting_tool import plot_centrality_distribution, plot_degree_distribution
import networkx as nx
import numpy as np


def analyze_individual_unweighted_networks(analyze_topology_and_GINI=0,
                                           analyze_node_metric_distribution=0,
                                           analyze_rb=0,
                                           analyze_relocation=1,
                                           all_modes_put_together=0):
    """
    Run analysis
    :param analyze_topology_and_GINI: True/False
    :param analyze_node_metric_distribution: True/False
    :param analyze_rb: True/False
    :param analyze_relocation: True/False
    :param all_modes_put_together: simply aggregation (without interconnection)
    :return: N/A
    """
    def check_stop_label(print_items=False):
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
    if not all_modes_put_together:
        steps = [0, 1, 2, 3, 4, 5]
    else:
        steps = [0]
    for i in steps:
        # import mptn network model
        mptn = create_mptn_model()
        if not all_modes_put_together:
            mode = all_modes[i]
            modes_to_remove = [item for item in all_modes if item != mode]
            for item in modes_to_remove:
                for subitem in item:
                    mptn.network.remove_routes_by_agency(subitem)
        check_stop_label()
        mptn.update_graph_by_routes_data()
        mptn.network.show()

        if analyze_topology_and_GINI:
            v, e = mptn.G.number_of_nodes(), mptn.G.number_of_edges()
            nd = round(mptn.preparedness_node_degree_gini(), 4)
            bc = round(mptn.preparedness_node_betweenness_centrality_gini(), 4)
            from benchmark_ER_expanding_unweighted_network import geospatial_efficiency, edge_length_distribution
            node_coordinates = {id: stop.coordinates() for id, stop in mptn.network.stop_repository.items()}
            ael, stdel, samples = edge_length_distribution(mptn.G, node_coordinates)
            from networkx import strongly_connected_components
            gcc = sorted(strongly_connected_components(mptn.G), key=len, reverse=True)
            n_g0 = mptn.G.subgraph(gcc[0]).number_of_nodes()
            prop = [['|V| = ', v],
                    ['|E| = ', e],
                    ['<k>', round(e / v, 2)],
                    ['S_0', n_g0 / mptn.G.number_of_nodes()],
                    ['l_max', xc.maximal_shortest_path_length(mptn.G)],
                    ['<l>', round(xc.average_shortest_path_length_modified_for_unconnected_digragh(mptn.G), 2)],
                    ['E', round(xc.global_efficiency_modified_for_unconnected_digragh(mptn.G), 2)],
                    ['GE', round(geospatial_efficiency(mptn.G, node_coordinates), 2)],
                    ['<l_e>', round(ael)],
                    ['std_l_e', round(stdel)],
                    ['GINI_nd =', round(nd, 3)],
                    ['GINI_bc =', round(bc, 3)]]
            for p in prop:
                print(p[0], ',', p[1])

        if analyze_node_metric_distribution:
            plot_centrality_distribution(mptn,
                                         save_figure_as=f'mptn_analyze_individual_unweighted_network_results/fig_centrality_distribution_{names[i]}.png', )
            plot_degree_distribution(mptn,
                                     save_figure_as=f'mptn_analyze_individual_unweighted_network_results/fig_degree_distribution_{names[i]}.png')

        if analyze_rb:
            rb_n_steps = 40
            rb_step_size = round(mptn.G.number_of_nodes() / rb_n_steps)
            xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=1000,
                                                                    multiple_removal=rb_step_size,
                                                                    multi_processing=True),
                           filename=f'mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_rnd.csv')
            xc.export_list(mptn.robustness_unweighted_degree_based_attack(number_of_tests=100,
                                                                          multiple_removal=rb_step_size,
                                                                          multi_processing=True),
                           filename=f'mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_nd.csv')
            xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(number_of_tests=100,
                                                                               multiple_removal=rb_step_size,
                                                                               multi_processing=True),
                           filename=f'mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_bc.csv')

        if analyze_relocation:
            if not all_modes_put_together:
                relocation_potential = mptn.path_based_unweighted_relocation(d_max=750)
                relos = [rl for node, rl in relocation_potential.items()]
                relo = np.mean(relos)
                print('average relocation =', round(float(relo), 3))
            else:
                relocation_potential = mptn.path_based_unweighted_relocation(d_max=750)
                for name in names:
                    relos = [rl for node, rl in relocation_potential.items() if
                             mptn.network.stop_repository[node].label == name]
                    print(name, len(relos))
                    relo = np.mean(relos)
                    print('average relocation =', round(float(relo), 3))


if __name__ == "__main__":
    analyze_individual_unweighted_networks(analyze_topology_and_GINI=1,
                                           analyze_node_metric_distribution=1,
                                           analyze_rb=1,
                                           analyze_relocation=1,
                                           all_modes_put_together=0)