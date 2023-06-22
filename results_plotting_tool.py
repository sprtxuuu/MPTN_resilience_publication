from xc_resilience_live import numerical_integral_nml
import matplotlib
from matplotlib import pyplot as plt
import csv
import numpy as np

rc = {"font.family": "serif",
      "mathtext.fontset": "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times"] + plt.rcParams["font.serif"]
matplotlib.rcParams.update({'font.size': 13})


def read_file(path_to_file, encode='utf-8-sig'):  # updated
    with open(path_to_file, 'r', encoding=encode) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))
        curves = []
        for row in data:
            curves.append([float(i) for i in row])
        return curves


def plot_rb_curves(path_to_files, fig_name, curves_name=None):
    def plot_curves(curves, fig_name='fig_curve.png', curves_name=None, p=1):
        figure = plt.figure(figsize=(4.2,3.5))
        ax0 = figure.add_subplot(1, 1, 1)
        axis_labels = ['$\\bf{c}$', '$\\bf{S}$']
        markers = ['o', 'x', '^', '+', 's', 'D']
        colors = ['b'] + ['r'] + ['g'] + ['k'] + ['c'] + ['m']
        linestyles = ['-', '--', '--', ':', ':', '-.']
        for i, curve in enumerate(curves):
            if p == 1:
                rbx = curve[1]
                rby = curve[0]
                if curves_name:
                    if len(curves_name) == len(curves):
                        label = f'{curves_name[i]} ($r_b={numerical_integral_nml(rby,xs=rbx, dec=3):.3f}$)'
                    else:
                        print('curves_name AND curves NEED IDENTICAL DIMENSION')
                        label = None
                else:
                    label = f'($r_b={numerical_integral_nml(rby, xs=rbx, dec=3):.3f}$)'
                ax0.plot(rbx, rby, label=label, color='grey', linewidth=0.5,
                         marker=markers[i], alpha=0.8, mfc='none', mec=colors[i],
                         markersize=5, markeredgewidth=1)
        ax0.legend()
        ax0.set_xlabel(axis_labels[0], fontsize=18)
        ax0.set_ylabel(axis_labels[1], fontsize=18)
        plt.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.94, wspace=0.33, hspace=0.4)
        # plt.show()
        plt.savefig(fig_name, transparent=True, dpi=450)

    rb_curves = [read_file(path) for path in path_to_files]
    plot_curves(rb_curves, fig_name=fig_name, curves_name=curves_name)


def plot_centrality_distribution(resilience_framework, save_figure_as='fig_centrality_distribution.png'):
    figure = plt.figure(figsize=(6, 4))
    centralities_dict = resilience_framework.get_node_betweenness_centrality()
    censeq = list(centralities_dict.values())
    cmax, dec = 1, 3  # cmax = round(max(censeq) + 0.1, 0)
    freq = [0 for c in np.arange(0, cmax, 0.1 ** dec)]
    for c in censeq:
        bin = round(c, dec)
        freq[int(bin * (10 ** dec))] += 1
    plt.figure(figsize=(6, 4))
    plt.loglog(np.arange(0, cmax, 0.1 ** dec), freq, 'ko', fillstyle="none", label='$c_B$')
    plt.xlabel('Betweenness centrality $c_B$')
    plt.ylabel('Frequency')
    plt.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.94, wspace=0.33, hspace=0.4)
    plt.savefig(save_figure_as, transparent=True, dpi=450)


def plot_degree_distribution(resilience_framework, save_figure_as='fig_degree_distribution.png'):
    figure = plt.figure(figsize=(6, 4))
    degree_dict = resilience_framework.get_node_degree()
    degseq = list(degree_dict.values())
    cmax = max(degseq)
    freq = [0 for c in np.arange(0, cmax + 1, 1)]
    for c in degseq:
        bin = c
        freq[int(bin)] += 1
    plt.figure(figsize=(6, 4))
    plt.loglog(np.arange(0, cmax + 1, 1), freq, 'ko', fillstyle="none", label='$k$')
    plt.xlabel('Node degree $k$')
    plt.ylabel('Frequency')
    plt.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.94, wspace=0.33, hspace=0.4)
    plt.savefig(save_figure_as, transparent=True, dpi=450)


def plot_distribution(data, file_name='fig_distribution.png',
                      xlabel='Value', ylabel='Frequency',
                      precision=1.0, loglog=True):
    figure = plt.figure(figsize=(6, 4))
    dataseq = list(data)
    cmax = max(dataseq)
    freq = [0 for c in np.arange(0, cmax + precision, precision)]
    for c in dataseq:
        bin = round(c / precision)
        freq[int(bin)] += 1
    plt.figure(figsize=(6, 4))
    if loglog:
        plt.loglog(np.arange(0, cmax + precision, precision), np.asarray(freq), 'ko', fillstyle="none",
                   label=f'Bin interval = {precision}')
    else:
        plt.scatter(np.arange(0, cmax + precision, precision), np.asarray(freq),
                    facecolors='none', edgecolors='k',
                    linewidths=1, label=f'Bin interval = {precision}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend(loc='lower left')
    plt.legend(loc='upper right')
    plt.subplots_adjust(left=0.20, bottom=0.16, right=0.96, top=0.94, wspace=0.33, hspace=0.4)
    plt.savefig(file_name, transparent=True, dpi=450)


def plot_scatter(xs, ys, file_name='fig_scatter.png',
                 xlabel='X', ylabel='Y'):
    figure = plt.figure(figsize=(6, 4))
    plt.scatter(np.array(xs), np.array(ys), s=10,
                # c='k',
                facecolors='none', edgecolors='k',
                linewidths=0.5,
                )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplots_adjust(left=0.18, bottom=0.16, right=0.96, top=0.94, wspace=0.33, hspace=0.4)
    plt.savefig(file_name, transparent=True, dpi=450)


def plot_for_expanding_unweighted_network_results_imt_0():
    base_path = 'mptn_analyze_expanding_unweighted_network_results'
    plot_rb_curves([base_path+'/mptn_imt_0_expanding_0_rb_rnd.csv',
                    base_path+'/mptn_imt_0_expanding_1_rb_rnd.csv',
                    base_path+'/mptn_imt_0_expanding_2_rb_rnd.csv',
                    base_path+'/mptn_imt_0_expanding_3_rb_rnd.csv',
                    base_path+'/mptn_imt_0_expanding_4_rb_rnd.csv',
                    base_path+'/mptn_imt_0_expanding_5_rb_rnd.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_0_random_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])

    plot_rb_curves([base_path+'/mptn_imt_0_expanding_0_rb_nd.csv',
                    base_path+'/mptn_imt_0_expanding_1_rb_nd.csv',
                    base_path+'/mptn_imt_0_expanding_2_rb_nd.csv',
                    base_path+'/mptn_imt_0_expanding_3_rb_nd.csv',
                    base_path+'/mptn_imt_0_expanding_4_rb_nd.csv',
                    base_path+'/mptn_imt_0_expanding_5_rb_nd.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_0_degree_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])

    plot_rb_curves([base_path+'/mptn_imt_0_expanding_0_rb_bc.csv',
                    base_path+'/mptn_imt_0_expanding_1_rb_bc.csv',
                    base_path+'/mptn_imt_0_expanding_2_rb_bc.csv',
                    base_path+'/mptn_imt_0_expanding_3_rb_bc.csv',
                    base_path+'/mptn_imt_0_expanding_4_rb_bc.csv',
                    base_path+'/mptn_imt_0_expanding_5_rb_bc.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_0_betweenness_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])


def plot_for_expanding_unweighted_network_results_imt_100():
    base_path = 'mptn_analyze_expanding_unweighted_network_results'
    plot_rb_curves([base_path+'/mptn_imt_100_expanding_0_rb_rnd.csv',
                    base_path+'/mptn_imt_100_expanding_1_rb_rnd.csv',
                    base_path+'/mptn_imt_100_expanding_2_rb_rnd.csv',
                    base_path+'/mptn_imt_100_expanding_3_rb_rnd.csv',
                    base_path+'/mptn_imt_100_expanding_4_rb_rnd.csv',
                    base_path+'/mptn_imt_100_expanding_5_rb_rnd.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_100_random_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])

    plot_rb_curves([base_path+'/mptn_imt_100_expanding_0_rb_nd.csv',
                    base_path+'/mptn_imt_100_expanding_1_rb_nd.csv',
                    base_path+'/mptn_imt_100_expanding_2_rb_nd.csv',
                    base_path+'/mptn_imt_100_expanding_3_rb_nd.csv',
                    base_path+'/mptn_imt_100_expanding_4_rb_nd.csv',
                    base_path+'/mptn_imt_100_expanding_5_rb_nd.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_100_degree_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])

    plot_rb_curves([base_path+'/mptn_imt_100_expanding_0_rb_bc.csv',
                    base_path+'/mptn_imt_100_expanding_1_rb_bc.csv',
                    base_path+'/mptn_imt_100_expanding_2_rb_bc.csv',
                    base_path+'/mptn_imt_100_expanding_3_rb_bc.csv',
                    base_path+'/mptn_imt_100_expanding_4_rb_bc.csv',
                    base_path+'/mptn_imt_100_expanding_5_rb_bc.csv'],
                   fig_name=base_path+'/fig_curve_mptn_imt_100_betweenness_attack.png',
                   curves_name=['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5', 'Step 6'])


def plot_for_individual_unweighted_network_results():
    plot_rb_curves(['mptn_analyze_individual_unweighted_network_results/mptn_MTR_rb_rnd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FB_rb_rnd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_GMB_rb_rnd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_LR_rb_rnd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FERRY_rb_rnd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_TRAM_rb_rnd.csv'],
                   fig_name='mptn_analyze_individual_unweighted_network_results/fig_curve_random_attack.png',
                   curves_name=['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM'])

    plot_rb_curves(['mptn_analyze_individual_unweighted_network_results/mptn_MTR_rb_nd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FB_rb_nd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_GMB_rb_nd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_LR_rb_nd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FERRY_rb_nd.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_TRAM_rb_nd.csv'],
                   fig_name='mptn_analyze_individual_unweighted_network_results/fig_curve_degree_attack.png',
                   curves_name=['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM'])

    plot_rb_curves(['mptn_analyze_individual_unweighted_network_results/mptn_MTR_rb_bc.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FB_rb_bc.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_GMB_rb_bc.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_LR_rb_bc.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_FERRY_rb_bc.csv',
                    'mptn_analyze_individual_unweighted_network_results/mptn_TRAM_rb_bc.csv'],
                   fig_name='mptn_analyze_individual_unweighted_network_results/fig_curve_betweenness_attack.png',
                   curves_name=['MTR', 'FB', 'GMB', 'LR', 'FERRY', 'TRAM'])


if __name__ == '__main__':
    plot_for_expanding_unweighted_network_results_imt_0()
    plot_for_expanding_unweighted_network_results_imt_100()
    plot_for_individual_unweighted_network_results()