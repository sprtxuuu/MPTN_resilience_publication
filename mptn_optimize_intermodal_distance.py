import toolbox
import xc_resilience_live as xc
from mptn_modeling_from_gtfs import create_mptn_model


def rb_value(file_path):
    from xc_resilience_live import numerical_integral_nml
    ys, xs, stds = toolbox.import_list(file_path=file_path, change_type='float')
    rb = numerical_integral_nml(ys, xs=xs)
    return rb


def run_simulation():
    def check_stop_label(print_items=False):
        i = 0
        for stop in mptn.network.stop_repository.values():
            if stop.label is None:
                stop.show()
            if print_items:
                print(stop.label)

    # log = [['IMT Distance', 'Robustness', 'Number of IMT edges', 'Number of edges']]
    dst_range = list(range(0, 400, 50)) + list(range(400, 1700, 100))
    print('D_IMT=', dst_range)
    for dst in dst_range:
        # import mptn network model
        mptn = create_mptn_model()
        # mptn.network.show()
        check_stop_label()
        # print(mptn.G.number_of_nodes(), mptn.G.number_of_edges())
        cache_dict = 'mptn_optimize_intermodal_distance_results/neighbor'
        intermodal_edges = mptn.network.generate_intermodal_edges(dst_limit=dst,
                                                                  path_to_save_neighbor_dict=cache_dict,
                                                                  read_previous_neighbor_dict=True)
        mptn.update_graph_by_routes_data(intermodal_edge_list=intermodal_edges)
        print('|V|=', mptn.G.number_of_nodes(), '|E|=', mptn.G.number_of_edges())
        print(len(mptn.network.functional_stop_list()))

        # visualization
        # mptn.plot(show=True)
        rb_n_steps = 40
        rb_step_size = round(mptn.G.number_of_nodes() / rb_n_steps)
        xc.export_list(mptn.robustness_unweighted_random_attack(number_of_tests=1000,
                                                                multiple_removal=rb_step_size,
                                                                multi_processing=True),
                       filename=f'mptn_optimize_intermodal_distance_results/optimization_rnd_dst_{dst}.csv')
        xc.export_list(mptn.robustness_unweighted_degree_based_attack(number_of_tests=100,
                                                                      multiple_removal=rb_step_size,
                                                                      multi_processing=True),
                       filename=f'mptn_optimize_intermodal_distance_results/optimization_nd_dst_{dst}.csv')
        xc.export_list(mptn.robustness_unweighted_betweenness_based_attack(number_of_tests=100,
                                                                           multiple_removal=rb_step_size,
                                                                           multi_processing=True),
                       filename=f'mptn_optimize_intermodal_distance_results/optimization_bc_dst_{dst}.csv')


def process_results_for_jmp(type=1):
    if type == 1:
        output_for_jmp = [['IMT Distance', 'Variable', 'Robustness', 'Number of IMT edges', 'Number of edges']]
        dst_range = list(range(0, 400, 50)) + list(range(400, 1700, 100))
        print('D_IMT=', dst_range)
        for dst in dst_range:
            mptn = create_mptn_model()
            print('|V|=', mptn.G.number_of_nodes(), '|E|=', mptn.G.number_of_edges())
            cache_dict = 'mptn_optimize_intermodal_distance_results/neighbor'
            intermodal_edges = mptn.network.generate_intermodal_edges(dst_limit=dst,
                                                                      path_to_save_neighbor_dict=cache_dict,
                                                                      read_previous_neighbor_dict=True)
            mptn.update_graph_by_routes_data(intermodal_edge_list=intermodal_edges)
            for scenario in ['rnd', 'nd', 'bc']:
                rb = rb_value(f'mptn_optimize_intermodal_distance_results/optimization_{scenario}_dst_{dst}.csv')
                output_for_jmp.append([dst, f'{scenario.upper()}', rb, len(intermodal_edges), mptn.G.number_of_edges()])
        toolbox.export_list(output_for_jmp, 'mptn_optimize_intermodal_distance_results/optimization_all.csv')

    if type == 2:
        output_for_jmp = [['s', 'c', 'Standard deviation', 'D_IMT']]
        dst_range = list(range(0, 400, 50)) + list(range(400, 1700, 100))
        for i in dst_range:
            curve = toolbox.import_list(f'mptn_optimize_intermodal_distance_results/optimization_dst_{i}.csv',
                                        change_type='float')

            ys, xs, stds = curve
            for k in range(len(ys)):
                output_for_jmp.append([ys[k], xs[k], stds[k], i])
        toolbox.export_list(output_for_jmp, 'mptn_optimize_intermodal_distance_results/optimization_rb_curves.csv')


run_simulation()
process_results_for_jmp(1)
