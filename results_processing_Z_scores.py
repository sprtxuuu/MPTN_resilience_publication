import csv
import toolbox as tb
import numpy as np

def data_process(file_path):
    title = tb.import_list(file_path=file_path)[0][1:]
    s = tb.import_list(file_path=file_path)[1][1:]
    s = np.asarray([float(item) for item in s])
    s = [1-v if i == 2 else v for i, v in enumerate(s)] # 1-gini
    b = tb.import_list(file_path=file_path)[2:]
    b = [[float(item) for item in row[1:]] for row in b]
    b = [[1-v if i == 2 else v for i, v in enumerate(row)] for row in b]
    std = np.std(b, axis=0)
    b = np.mean(b, axis=0)
    z = [s[i] - b[i] if s[i] - b[i] == 0 or std[i] == 0 else round((s[i] - b[i])/std[i],3) for i in range(len(s))]
    # z = [(s[i] - b[i])/b[i] for i in range(len(s))]
    return title, z


def rb_value(file_path):
    from xc_resilience_live import numerical_integral_nml
    ys, xs, stds = tb.import_list(file_path=file_path, change_type='float')
    rb = numerical_integral_nml(ys, xs=xs)
    return rb


def generate_rb_curve_all_in_one():
    def rb_curve(file_path, labels):
        ys, xs, stds = tb.import_list(file_path=file_path, change_type='float')
        points = []
        for i in range(len(ys)):
            points.append([ys[i], xs[i]] + labels)
        return points
    ### add benchmark
    all_in_one = [['s', 'c', 'step', 'test', 'scenario', 'group']]
    for scenario in ['rnd', 'nd', 'bc']:
        for step in range(6):
            for test in range(10):
                labels = [str(int(step)+1), str(test), str(scenario).upper(), 'Benchmark']
                all_in_one.extend(rb_curve(f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{step}_rb_{scenario}_{test}.csv',
                                  labels=labels))
    ### add system
    root = 'mptn_analyze_expanding_unweighted_network_results/'
    for scenario in ['rnd', 'nd', 'bc']:
        for step in range(6):
            file_path = root + f'mptn_imt_100_expanding_{step}_rb_{scenario}.csv'
            labels = [str(int(step)+1), '0', str(scenario).upper(), 'System']
            all_in_one.extend(rb_curve(file_path, labels))
    return all_in_one

# for ptn in ['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM']:
#     print(ptn, data_process(f'{ptn}_geospatial_efficiency.csv'))


def show_statistics(type):
    if type == 1:
        for imt in [0, 1]:
            for step in range(6):
                title, z_score = data_process(f'Benchmark_gs_ER_results/expanding_imt_{imt}_step_{step}_geospatial_efficiency.csv')
                if step == 0:
                    print('step', 'i', title, f'imt={imt}')
                print('step', step, z_score)
    if type == 2:
        for imt in [0, 1]:
            print(f'Step, Indicator, Z-score, imt={imt}')
            for step in range(6):
                title, z_score = data_process(f'Benchmark_gs_ER_results/expanding_imt_{imt}_step_{step}_geospatial_efficiency.csv')
                for i, v in enumerate(z_score):
                    if i not in [0, 1]:
                        print(f'{step}, {title[i]}, {v}')

def imt_100_z_score_rb(num_of_tests_1, num_of_tests_2, num_of_tests_3):
    ss_folder = 'mptn_analyze_expanding_unweighted_network_results'
    print(f'Step, Indicator, Z-score, IMT Distance')
    for step in range(6):
        # ss = [0.197, 0.198, 0.300, 0.287, 0.288, 0.290]
        s = rb_value(f'{ss_folder}/mptn_imt_100_expanding_{step}_rb_rnd.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{step}_rb_rnd_{test}.csv') for test in range(num_of_tests_1)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(RND), {round((s-b)/std, 3)}, D_IMT=100')

    for step in range(6):
        # ss = [0.083, 0.085, 0.156, 0.151, 0.152, 0.156]
        s = rb_value(f'{ss_folder}/mptn_imt_100_expanding_{step}_rb_nd.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{step}_rb_nd_{test}.csv') for test in range(num_of_tests_2)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(ND), {round((s - b) / std, 3)}, D_IMT=100')

    for step in range(6):
        # ss = [0.104, 0.102, 0.131, 0.125, 0.125, 0.124]
        s = rb_value(f'{ss_folder}/mptn_imt_100_expanding_{step}_rb_bc.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_100_expanding_{step}_rb_bc_{test}.csv') for test in range(num_of_tests_3)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(BC), {round((s - b) / std, 3)}, D_IMT=100')


def imt_0_z_score_rb(num_of_tests_1, num_of_tests_2, num_of_tests_3):
    ss_folder = 'mptn_analyze_expanding_unweighted_network_results'
    print(f'Step, Indicator, Z-score, IMT Distance')
    for step in range(6):
        # ss = [0.200, 0.188, 0.097, 0.092, 0.092, 0.090]
        s = rb_value(f'{ss_folder}/mptn_imt_0_expanding_{step}_rb_rnd.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{step}_rb_rnd_{test}.csv') for test in range(num_of_tests_1)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(RND), {round((s-b)/std, 3)}, D_IMT=0')

    for step in range(6):
        # ss = [0.082, 0.072, 0.030, 0.028, 0.028, 0.028]
        s = rb_value(f'{ss_folder}/mptn_imt_0_expanding_{step}_rb_nd.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{step}_rb_nd_{test}.csv') for test in range(num_of_tests_2)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(ND), {round((s - b) / std, 3)}, D_IMT=0')

    for step in range(6):
        # ss = [0.104, 0.089, 0.047, 0.043, 0.043, 0.431]
        s = rb_value(f'{ss_folder}/mptn_imt_0_expanding_{step}_rb_bc.csv')
        rbs = [rb_value(f'Benchmark_gs_ER_results/robustness/mptn_imt_0_expanding_{step}_rb_bc_{test}.csv') for test in range(num_of_tests_3)]
        b, std = np.mean(rbs), np.std(rbs)
        # s = ss[step]
        print(f'{step}, rb(BC), {round((s - b) / std, 3)}, D_IMT=0')


if __name__ == '__main__':
    imt_100_z_score_rb(10, 10, 10)
    imt_0_z_score_rb(10, 10, 10)
    show_statistics(2)
    # tb.export_list(generate_rb_curve_all_in_one(), 'Benchmark_gs_ER_results/rb_curve_all_in_one.csv')