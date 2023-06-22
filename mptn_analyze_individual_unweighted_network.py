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
    # for i in range(6):
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
            # print(mptn.G.number_of_nodes(), mptn.G.number_of_edges(), len(mptn.network.functional_stop_list()))
            for item in modes_to_remove:
                for subitem in item:
                    mptn.network.remove_routes_by_agency(subitem)
        check_stop_label()
        mptn.update_graph_by_routes_data()
        mptn.network.show()
        # mptn.plot(show=True)

        if analyze_topology_and_GINI:  # outdated
            print('|V| = ', mptn.G.number_of_nodes())
            print('|E| = ', mptn.G.number_of_edges())
            print('GINI_nd =', round(mptn.preparedness_node_degree_gini(), 4))
            print('GINI_bc =', round(mptn.preparedness_node_betweenness_centrality_gini(), 4))
            print('CPL=', round(nx.average_shortest_path_length(mptn.G), 4))
            print('radius=', nx.radius(mptn.G))
            print('E=', nx.global_efficiency(mptn.G))

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
                           filename=f'new_mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_rnd.csv')
            xc.export_list(mptn.robustness_unweighted_degree_based_attack(number_of_tests=100,
                                                                          multiple_removal=rb_step_size,
                                                                          multi_processing=True),
                           filename=f'new_mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_nd.csv')
            xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(number_of_tests=100,
                                                                               multiple_removal=rb_step_size,
                                                                               multi_processing=True),
                           filename=f'new_mptn_analyze_individual_unweighted_network_results/mptn_{names[i]}_rb_bc.csv')

        if analyze_relocation:
            if not all_modes_put_together:
                relocation_potential = mptn.path_based_unweighted_relocation(d_max=750)
                relos = [rl for node, rl in relocation_potential.items()]
                relo = np.mean(relos)
                print('average relocation =', round(float(relo), 3))
            else:
                relocation_potential = mptn.path_based_unweighted_relocation(d_max=750)
                # print(np.mean([rl for node, rl in relocation_potential.items()]))
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