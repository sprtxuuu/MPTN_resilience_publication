import xc_resilience_live as xc
from mptn_modeling_from_gtfs import create_mptn_model
from results_plotting_tool import plot_centrality_distribution, plot_degree_distribution
import numpy as np
from toolbox import export_list


def analyze_expanding_unweighted_network(analyze_topology_and_GINI=0,
                                         analyze_node_metric_distribution=0,
                                         analyze_rb=0,
                                         imt_edges=0,
                                         analyze_relocation=0,
                                         d_max=1600,
                                         complete_mptn_only=0):
    """
    run analysis
    :param analyze_topology_and_GINI: True/False
    :param analyze_node_metric_distribution: True/False
    :param analyze_rb: True/False
    :param imt_edges: True/False, enable intermodal edges (interconnected or isolated MPTN)
    :param analyze_relocation: True/False
    :param d_max: int, maximum relocation distance, 0-1600 meters
    :param complete_mptn_only: True/False, only run analysis for complete MPTN (skip analysis during integration)
    :return: N/A
    """
    def check_stop_label(mptn, print_items=False):
        i = 0
        for stop in mptn.network.stop_repository.values():
            if stop.label is None:
                stop.show()
            if print_items:
                print(stop.label)

    def export_attribute_in_stop_list(mptn, attribute_dict, path_to_file):
        table = [['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'zone_id', 'location_type', 'stop_timezone',
                  'label', 'agency_id', 'attribute']]
        for stop_id in mptn.network.functional_stop_list():
            stop = mptn.network.stop_repository[stop_id]
            table.append([stop.stop_id, stop.stop_name, stop.stop_lat, stop.stop_lon, stop.zone_id, stop.location_type,
                          stop.stop_timezone, stop.label, stop.agency_id, relocation_potential[stop_id]])
        export_list(table, path_to_file)


    all_modes = [['MTR'],
                 ['CTB', 'KMB+CTB', 'KMB', 'NLB', 'PI', 'LWB', 'DB', 'LRTFeeder', 'KMB+NWFB', 'LWB+CTB', 'NWFB', 'XB'],
                 ['GMB'],
                 ['LR'],
                 ['FERRY'],
                 ['TRAM']]

    names = ['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM']
    log = []

    for i in range(6):
        mode = all_modes[:i + 1]
        print(mode)
        modes_to_remove = [item for item in all_modes if item not in mode]
        print(modes_to_remove)
        # import mptn network model
        mptn = create_mptn_model()
        check_stop_label(mptn)
        for item in modes_to_remove:
            for subitem in item:
                mptn.network.remove_routes_by_agency(subitem)

        if imt_edges:
            intermodal_edges = mptn.network.generate_intermodal_edges(dst_limit=100)
            mptn.update_graph_by_routes_data(intermodal_edge_list=intermodal_edges)
        else:
            mptn.update_graph_by_routes_data()
        mptn.network.show()

        if analyze_topology_and_GINI:
            if (not complete_mptn_only) or (complete_mptn_only and i == 5):
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
            if (not complete_mptn_only) or (complete_mptn_only and i == 5):
                plot_centrality_distribution(mptn,
                                             save_figure_as=f'mptn_analyze_expanding_unweighted_network_results/fig_centrality_distribution_{i}.png', )
                plot_degree_distribution(mptn,
                                         save_figure_as=f'mptn_analyze_expanding_unweighted_network_results/fig_degree_distribution_{i}.png')

        if analyze_rb:
            if (not complete_mptn_only) or (complete_mptn_only and i == 5):
                rb_n_steps = 20
                rb_step_size = round(mptn.G.number_of_nodes() / rb_n_steps)
                if imt_edges:
                    xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=1000,
                                                                            multiple_removal=rb_step_size,
                                                                            multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_100_expanding_{i}_rb_rnd.csv')
                    xc.export_list(mptn.robustness_unweighted_degree_based_attack(number_of_tests=100,
                                                                                  multiple_removal=rb_step_size,
                                                                                  multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_100_expanding_{i}_rb_nd.csv')
                    xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(number_of_tests=100,
                                                                                       multiple_removal=rb_step_size,
                                                                                       multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_100_expanding_{i}_rb_bc.csv')
                else:
                    xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=1000,
                                                                            multiple_removal=rb_step_size,
                                                                            multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_0_expanding_{i}_rb_rnd.csv')
                    xc.export_list(mptn.robustness_unweighted_degree_based_attack(number_of_tests=100,
                                                                                  multiple_removal=rb_step_size,
                                                                                  multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_0_expanding_{i}_rb_nd.csv')
                    xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(number_of_tests=100,
                                                                                       multiple_removal=rb_step_size,
                                                                                       multi_processing=True),
                                   filename=f'mptn_analyze_expanding_unweighted_network_results/mptn_imt_0_expanding_{i}_rb_bc.csv')
            print('\n')

        if analyze_relocation:
            if (not complete_mptn_only) or (complete_mptn_only and i == 5):
                relocation_potential = mptn.path_based_unweighted_relocation(d_max=d_max)
                relos = [rl for node, rl in relocation_potential.items()]
                if relos:
                    relo = np.mean(relos)
                else:
                    relo = 'nan'
                print('global average relocation =', round(float(relo), 3))
                for name in names:
                    relos = [rl for node, rl in relocation_potential.items() if
                             mptn.network.stop_repository[node].label == name]
                    print(name, len(relos))
                    if relos:
                        relo = np.mean(relos)
                    else:
                        relo = 'nan'
                    print('average relocation =', round(float(relo), 3))
                export_attribute_in_stop_list(mptn, attribute_dict=relocation_potential,
                                              path_to_file=f'mptn_analyze_expanding_unweighted_network_results/'
                                                           f'stop_relocation_IMT_{imt_edges}_relocation_{d_max}_step_{i}.csv')

    xc.export_list(log, 'mptn_analyze_expanding_unweighted_network_results/log.csv')
    # for dst in dst_range:
    #     intermodal_edges = mptn.network.generate_intermodal_edges(dst_limit=dst)
    #     mptn.update_graph_by_routes_data(intermodal_edge_list=intermodal_edges)
    #     print(mptn.G.number_of_nodes(), mptn.G.number_of_edges())
    #
    #     # print(len(mptn.network.stops), mptn.G.number_of_nodes(), len(mptn.network.functional_stop_list()))
    #
    #     # visualization
    #     # mptn.plot(show=True)
    #
    #     result = mptn.robustness_unweighted_random_attack(number_of_tests=100,
    #                                                       multiple_removal=100,
    #                                                       multi_processing=True)
    #     xc.export_list(result, filename=f'mptn_optimize_intermodal_distance_results/optimization_dst_{dst}.csv')
    #     rb = xc.numerical_integral_nml(result[0], result[1])
    #     log.append([dst, rb, len(intermodal_edges)])
    # print(log)
    # xc.export_list(log, 'mptn_optimize_intermodal_distance_results/optimization_curve.csv')


if __name__ == '__main__':
    analyze_expanding_unweighted_network(analyze_topology_and_GINI=1,
                                         analyze_node_metric_distribution=1,
                                         analyze_rb=1,
                                         imt_edges=0,
                                         analyze_relocation=0,
                                         d_max=750,
                                         complete_mptn_only=0)

    analyze_expanding_unweighted_network(analyze_topology_and_GINI=1,
                                         analyze_node_metric_distribution=1,
                                         analyze_rb=1,
                                         imt_edges=1,
                                         analyze_relocation=0,
                                         d_max=750,
                                         complete_mptn_only=0)

    # relocation analysis
    analyze_expanding_unweighted_network(analyze_topology_and_GINI=0,
                                         analyze_node_metric_distribution=0,
                                         analyze_rb=0,
                                         imt_edges=0,
                                         analyze_relocation=1,
                                         d_max=750,
                                         complete_mptn_only=0)

    analyze_expanding_unweighted_network(analyze_topology_and_GINI=0,
                                         analyze_node_metric_distribution=0,
                                         analyze_rb=0,
                                         imt_edges=1,
                                         analyze_relocation=1,
                                         d_max=750,
                                         complete_mptn_only=0)

    analyze_expanding_unweighted_network(analyze_topology_and_GINI=0,
                                         analyze_node_metric_distribution=0,
                                         analyze_rb=0,
                                         imt_edges=0,
                                         analyze_relocation=1,
                                         d_max=1600,
                                         complete_mptn_only=0)

    analyze_expanding_unweighted_network(analyze_topology_and_GINI=0,
                                         analyze_node_metric_distribution=0,
                                         analyze_rb=0,
                                         imt_edges=1,
                                         analyze_relocation=1,
                                         d_max=1600,
                                         complete_mptn_only=0)
