def multiprocess_function(function, args, number_of_tests, core_num=5):
    import multiprocessing
    pool = multiprocessing.Pool(processes=core_num)
    if args:
        pool_result = [pool.apply_async(function, args=args) for test in range(number_of_tests)]
    else:
        pool_result = [pool.apply_async(function) for test in range(number_of_tests)]
    pool.close()
    pool.join()
    results = []
    for nt in pool_result:
        results.append(nt.get())
    return results


def divide_sequential_edge_list(edge_list):
    if len(edge_list) == 1:
        sections = [edge_list]
    elif len(edge_list) > 1:
        sections = []
        section = [edge_list[0]]
        for i in range(1, len(edge_list)):
            if edge_list[i][0] == edge_list[i - 1][1]:
                section.append(edge_list[i])
            else:
                sections.append(section)
                section = [edge_list[i]]
        sections.append(section)
    else:
        sections = None
    return sections


def remove_duplicate(a):
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(a))


def print_dict(dic, num):
    i_num = 0
    for key, values in dic.items():
        if i_num > num:
            break
        print(key, values)
        i_num += 1


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


def export_list(xp_list, filename, mode='w', contain_chinese=False):
    import csv
    if contain_chinese:
        encode = 'utf-8-sig'
    else:
        encode = 'utf-8'
    with open(filename, mode, encoding=encode, newline='') as output_file:
        # print(xp_list)
        csv_writer = csv.writer(output_file, delimiter=',')
        for x in xp_list:
            csv_writer.writerow(x)
    print('file:', filename, 'created')


def import_list(file_path, change_type=None):
    import csv
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ip_list = []
        for row in csv_reader:
            if change_type == 'float':
                row = [float(item) for item in row]
            elif change_type == 'int':
                row = [int(item) for item in row]
            ip_list.append(row)
    return ip_list


def save_pet(pet, filename='temporary file'):
    import json
    with open(filename, 'w') as f:
        f.write(json.dumps(str(pet)))


def load_pet(filename):
    import json
    with open(filename) as f:
        pet = json.loads(f.read())
    return eval(pet)


def revert_dict_of_list(original_dict):
    from collections import defaultdict
    """
    :param original_dict: a dict of lists
    """
    new_dict = defaultdict(list)
    for key, value in original_dict.items():
        for v in value:
            new_dict[v].append(key)
    return dict(new_dict)


def print_dataframe(dataframe):
    import pandas
    with pandas.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.expand_frame_repr', False):
        print(dataframe)


def sort_dict_by_value(a_dict):
    return dict(sorted(a_dict.items(), key=lambda item: item[1]))


class Canvas:
    def __init__(self):
        self.curves = {}
        self.marker_list = ['o'] + ['^'] + ['s'] + ['x'] + ['+'] + ['h']
        self.color_list = ['k'] + ['b'] + ['darkgreen'] + ['r'] + ['darkorange'] + ['m'] + ['c'] + ['y']
        self.line_style_list = ['-'] + ['--'] + [':'] + ['-.']
        self.linewidth = 0.5
        self.markersize = 4
        self.alpha = 0.8
        self.legend_ncol = 2
        self.legend_size = 10.5
        self.adjust = {'left': 0.12, 'bottom': 0.16, 'right': 0.96, 'top': 0.92, 'wspace': 0.33, 'hspace': 0.4}
        self.dpi = 300

    def add_curve(self, xs, ys, label):
        self.curves[label] = [xs, ys]

    def plot_curves(self, list_of_curve_labels, x_axis_label=None, y_axis_label=None, show=True, save_path=None,
                    customize_label_in_figure=None):
        from matplotlib import pyplot as plt
        if x_axis_label is None:
            x_axis_label = 'x'
        if y_axis_label is None:
            y_axis_label = 'y'
        figure = plt.figure(figsize=(6, 4))
        ax0 = figure.add_subplot(1, 1, 1)
        for i, curve_label in enumerate(list_of_curve_labels):
            xs, ys = self.curves[curve_label]
            if customize_label_in_figure:
                label_in_figure = customize_label_in_figure[curve_label]
            else:
                label_in_figure = curve_label
            ax0.plot(xs, ys, label=label_in_figure, color='grey',
                     marker=self.marker_list[i % len(self.marker_list)],
                     alpha=self.alpha,
                     mfc='none',
                     mec=self.color_list[i % len(self.color_list)],
                     markersize=self.markersize,
                     linewidth=self.linewidth)
        ax0.legend(ncol=self.legend_ncol, prop={'size': self.legend_size})
        ax0.set_xlabel(x_axis_label)
        ax0.set_ylabel(y_axis_label)
        plt.subplots_adjust(left=self.adjust['left'], bottom=self.adjust['bottom'], right=self.adjust['right'],
                            top=self.adjust['top'], wspace=self.adjust['wspace'], hspace=self.adjust['hspace'])
        if save_path:
            plt.savefig(save_path, transparent=True, dpi=self.dpi)
        if show:
            plt.show()
