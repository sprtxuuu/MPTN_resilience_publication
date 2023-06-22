import copy
import xc_resilience_live as xc
import pandas as pd
import numpy as np

'''
# weighted network
# vehicle capacity (person)
# #vehicle capacity: Upper Deck Seating: 63 Lower Deck Seating: 35 Lower Deck Standees: 48
# # 63+35+48 / double decker
# # http://www.kmb.hk/en/news/press/archives/news201408222062.html
# # 19 / minibus
# # 115 / tram
# # https://en.wikipedia.org/wiki/Hong_Kong_Tramways
# # 300, 388, 410, 170-200, 180, 100 / ferry
# # http://www.hkkf.com.hk/index.php?op=show&page=fleet&style=en
# # 3750, 2500/ train
# # https://www.mtr.com.hk/en/corporate/operations/detail_worldclass.html
# # 26 seated and 185 standees / 211 / light rail
'''


def create_mptn_model(peak_hour=0):
    """

    :param peak_hour: peak hour filter, default false
    :return: class Resilience (analysis framework)
    """
    # import GTFS dataset
    mptn = xc.Resilience('mptn')
    mptn.network.load_from_gtfs_dataset(folder_path='Raw database/GTFS_hk_31Jan2022', peak_hour=peak_hour)
    mptn.network.remove_routes_by_agency('GMB')
    mptn.network.remove_routes_by_agency('PTRAM')
    valid_stops = set()
    for route in mptn.network.routes.values():
        valid_stops = valid_stops.union(route.stop_set())
    invalid_stops = set(mptn.network.stop_repository_list()) - valid_stops
    for stop_id in invalid_stops:
        mptn.network.stop_repository.pop(stop_id)

    # add missing tram trips
    for route in mptn.network.routes.values():
        if route.agency_id == 'TRAM':
            # print(len(route.trip_list()))
            new_trips_dict = {}
            for trip in route.trips.values():
                for i in range(8):
                    new_trip = copy.deepcopy(trip)
                    new_trip.trip_id = trip.trip_id + '-supplemental-' + str(i)
                    new_trips_dict[new_trip.trip_id] = new_trip
            route.trips = new_trips_dict

    # add high quality GMB data
    stop_location = pd.read_csv('Raw database/GMB_high_quality_data/GMB gps.csv', index_col=2)
    stop_location = stop_location[['x', 'y']].to_dict(orient='index')
    stop_location = {key.lower(): value for key, value in stop_location.items()}
    # print(stop_location)
    file = 'Raw database/GMB_high_quality_data/complete_gmb_edge_frequency.csv'
    gmb = pd.read_csv(file)
    gmb_grp = gmb.groupby(by='route_name_seq_area')
    for route_id, item in gmb_grp:
        stop_seq_table = gmb_grp.get_group(route_id)
        # xc.print_dataframe(stop_seq_table)
        table = stop_seq_table.reset_index(drop=True)
        # xc.print_dataframe(table)
        table = table.to_dict(orient='index')
        mptn.network.add_route(route_id=route_id, agency_id='GMB',
                               route_long_name=table[0]['route_long_name'],
                               route_short_name=table[0]['route_short_name'],
                               route_url=table[0]['route_id'],
                               vehicle_capacity=19)
        freq_min, freq_max = table[0]['ti_min_weekday'], table[0]['ti_max_weekday']
        if isinstance(freq_min, float):
            if np.isnan(freq_min):
                freq_min = 0
        else:
            freq_min = float(freq_min)
            freq_min = int(60 / freq_min)

        if isinstance(freq_max, float):
            if np.isnan(freq_max):
                freq_max = 0
        else:
            freq_max = float(freq_max)
            freq_max = int(60 / freq_max)

        if peak_hour == -1:
            freq = 6 * (0 * freq_max + 1 * freq_min) / 1
        elif peak_hour == 1:
            freq = 6 * (1 * freq_max + 0 * freq_min) / 1
        else:
            freq = 12 * (0 * freq_max + 1 * freq_min) / 1  # same as non-peak
        edge_list = stop_seq_table[['stop_u', 'stop_v']].values.tolist()
        path = xc.remove_duplicate([edge[i] for edge in edge_list for i in range(2)])
        path = ['GMB-' + str(stop) for stop in path]
        edge_list = [(path[pl], path[pl + 1]) for pl in range(len(path) - 1)]
        for stop in path:
            original_name = stop[4:].lower()
            lat, lon = float(stop_location[original_name]['y']), float(stop_location[original_name]['x'])
            mptn.network.add_stop(stop_id=stop, stop_name=stop,
                                  stop_lat=lat, stop_lon=lon,
                                  label='GMB')
        mptn.network.routes[route_id].add_multiple_trips_by_frequency(trip_id_prefix=route_id,
                                                                      frequency=freq,
                                                                      service_id=None,
                                                                      edge_list=edge_list)
        # mptn.compute_edge_capacity()

    # add MTR
    stop_location = pd.read_csv('Raw database/MTR_27Jun2021/mtr_station_param.csv', index_col=0)
    stop_location = stop_location[['stop_lat', 'stop_lon']].to_dict(orient='index')
    if peak_hour == -1:
        freq_by_line_direction = {
            "AEL-DT": 2, "AEL-UT": 2,
            "DRL-DT": 4, "DRL-UT": 4,
            "EAL-DT": 5, "EAL-UT": 5,
            "EAL-LMC-DT": 4, "EAL-LMC-UT": 4,
            "ISL-DT": 10, "ISL-UT": 10,
            "KTL-DT": 10, "KTL-UT": 10,
            "SIL-DT": 8, "SIL-UT": 8,
            "TCL-DT": 6, "TCL-UT": 6,
            "TKL-DT": 10, "TKL-UT": 10,
            "TKL-TKS-DT": 5, "TKL-TKS-UT": 5,
            "TML-DT": 8, "TML-UT": 8,
            "TWL-DT": 10, "TWL-UT": 10}  # non-peak
    elif peak_hour == 1:
        freq_by_line_direction = {
            "AEL-DT": 4, "AEL-UT": 4,
            "DRL-DT": 6, "DRL-UT": 6,
            "EAL-DT": 5, "EAL-UT": 5,
            "EAL-LMC-DT": 5, "EAL-LMC-UT": 5,
            "ISL-DT": 30, "ISL-UT": 30,
            "KTL-DT": 30, "KTL-UT": 30,
            "SIL-DT": 18, "SIL-UT": 18,
            "TCL-DT": 10, "TCL-UT": 10,
            "TKL-DT": 24, "TKL-UT": 24,
            "TKL-TKS-DT": 9, "TKL-TKS-UT": 9,
            "TML-DT": 20, "TML-UT": 20,
            "TWL-DT": 30, "TWL-UT": 30}  # peak hour
    else:
        freq_by_line_direction = {
            "AEL-DT": 3, "AEL-UT": 3,
            "DRL-DT": 5, "DRL-UT": 5,
            "EAL-DT": 5, "EAL-UT": 5,
            "EAL-LMC-DT": 5, "EAL-LMC-UT": 5,
            "ISL-DT": 20, "ISL-UT": 20,
            "KTL-DT": 20, "KTL-UT": 20,
            "SIL-DT": 13, "SIL-UT": 13,
            "TCL-DT": 8, "TCL-UT": 8,
            "TKL-DT": 17, "TKL-UT": 17,
            "TKL-TKS-DT": 7, "TKL-TKS-UT": 7,
            "TML-DT": 14, "TML-UT": 14,
            "TWL-DT": 20, "TWL-UT": 20}  # mean

    file = 'Raw database/MTR_27Jun2021/mtr_lines_and_stations.csv'
    mtr = pd.read_csv(file)
    mtr_grp = mtr.groupby(by='Line-Direction')
    for line_direction, item in mtr_grp:
        stop_seq_table = mtr_grp.get_group(line_direction)
        # xc.print_dataframe(stop_seq_table)
        table = stop_seq_table.reset_index(drop=True)
        table = table.to_dict(orient='index')
        long_name = table[0]['English Name'] + ' - ' + table[max(table.keys())]['English Name']
        # print(line_direction, long_name)
        # print(f"\"{line_direction}\":12,")
        mptn.network.add_route(route_id=line_direction, agency_id='MTR',
                               route_long_name=long_name,
                               route_short_name=line_direction,
                               route_url=None,
                               vehicle_capacity=2500)
        path = stop_seq_table['Station Code'].values.tolist()
        path = ['MTR-' + str(stop) for stop in path]
        edge_list = [(path[pl], path[pl + 1]) for pl in range(len(path) - 1)]
        # print(path)
        for seq, stop in enumerate(path):
            stop_name = table[seq]['English Name']
            lat, lon = float(stop_location[stop_name]['stop_lat']), float(stop_location[stop_name]['stop_lon'])
            mptn.network.add_stop(stop_id=stop, stop_name=stop_name,
                                  stop_lat=lat, stop_lon=lon,
                                  label='MTR')
        if peak_hour:
            frequency = 6 * freq_by_line_direction[line_direction]
        else:
            frequency = 12 * freq_by_line_direction[line_direction]
        mptn.network.routes[line_direction].add_multiple_trips_by_frequency(trip_id_prefix=line_direction,
                                                                            frequency=frequency,
                                                                            service_id=None,
                                                                            edge_list=edge_list)

    # add Light Rail
    stop_location = pd.read_csv('Raw database/MTR_27Jun2021/light rail stops coordinates.csv', index_col=3)
    stop_location = stop_location[['MEAN_Y', 'MEAN_X']].to_dict(orient='index')
    # print(stop_location)
    if peak_hour == -1:
        freq_by_line_direction = {
            "505-1": 5, "505-2": 5,
            "507-1": 4, "507-2": 4,
            "610-1": 5, "610-2": 5,
            "614-1": 3, "614-2": 3,
            "614P-1": 4, "614P-2": 4,
            "615-1": 3, "615-2": 3,
            "615P-1": 4, "615P-2": 4,
            "705-1": 7, "705-2": 7,
            "706-1": 6, "706-2": 6,
            "751-1": 5, "751-2": 5,
            "761P-1": 6, "761P-2": 6}  # non-peak
    elif peak_hour == 1:
        freq_by_line_direction = {
            "505-1": 9, "505-2": 9,
            "507-1": 7, "507-2": 7,
            "610-1": 6, "610-2": 6,
            "614-1": 4, "614-2": 4,
            "614P-1": 6, "614P-2": 6,
            "615-1": 4, "615-2": 4,
            "615P-1": 6, "615P-2": 6,
            "705-1": 11, "705-2": 11,
            "706-1": 11, "706-2": 11,
            "751-1": 7, "751-2": 7,
            "761P-1": 9, "761P-2": 9}  # peak
    else:
        freq_by_line_direction = {
            "505-1": 7, "505-2": 7,
            "507-1": 6, "507-2": 6,
            "610-1": 6, "610-2": 6,
            "614-1": 4, "614-2": 4,
            "614P-1": 5, "614P-2": 5,
            "615-1": 4, "615-2": 4,
            "615P-1": 5, "615P-2": 5,
            "705-1": 9, "705-2": 9,
            "706-1": 9, "706-2": 9,
            "751-1": 6, "751-2": 6,
            "761P-1": 8, "761P-2": 8}  # mean
    file = 'Raw database/MTR_27Jun2021/light_rail_routes_and_stops.csv'
    lr = pd.read_csv(file)
    lr_grp = lr.groupby(by='Line-Direction')
    for line_direction, item in lr_grp:
        stop_seq_table = lr_grp.get_group(line_direction)
        # xc.print_dataframe(stop_seq_table)
        table = stop_seq_table.reset_index(drop=True)
        table = table.to_dict(orient='index')
        long_name = table[0]['English Name'] + ' - ' + table[max(table.keys())]['English Name']
        mptn.network.add_route(route_id=line_direction, agency_id='LR',
                               route_long_name=long_name,
                               route_short_name=line_direction,
                               route_url=None,
                               vehicle_capacity=211)
        path = stop_seq_table['Stop Code'].values.tolist()
        path = ['LR-' + str(stop) for stop in path]
        edge_list = [(path[pl], path[pl + 1]) for pl in range(len(path) - 1)]
        # print(path)
        for seq, stop in enumerate(path):
            stop_name = table[seq]['English Name']
            lat, lon = stop_location[stop_name]['MEAN_Y'], stop_location[stop_name]['MEAN_X']
            mptn.network.add_stop(stop_id=stop, stop_name=stop_name,
                                  stop_lat=lat, stop_lon=lon,
                                  label='LR')
        if peak_hour:
            frequency = 6 * freq_by_line_direction[line_direction]
        else:
            frequency = 12 * freq_by_line_direction[line_direction]
        mptn.network.routes[line_direction].add_multiple_trips_by_frequency(trip_id_prefix=line_direction,
                                                                            frequency=frequency,
                                                                            service_id=None,
                                                                            edge_list=edge_list)

    mptn.network.remove_empty_routes()
    mptn.update_graph_by_routes_data()
    return mptn


if __name__ == '__main__':
    """for test
    peak_hour=1 (peak time 6 hours: 7-10, 17-20)
    peak_hour=-1 (non-peak time)
    peak_hour=0/False (disable filter)
    """

    mptn = create_mptn_model(peak_hour=0)
    print(mptn.G.number_of_nodes())
    mptn.update_graph_by_routes_data(update_node_label=True)
    print(mptn.G.number_of_nodes())
    print(mptn.G.number_of_edges())
    print(mptn.network.compute_total_capacity())
    nroute, ntrip = 0, 0
    for route in mptn.network.routes.values():
        nroute+=1
        for trip in route.trips.values():
            ntrip+=1
    print(nroute, ntrip)
    mptn.network.show()
    # agency_list = {
    #     'FB': ['CTB', 'KMB+CTB', 'KMB', 'NLB', 'PI', 'LWB', 'DB', 'LRTFeeder', 'KMB+NWFB', 'LWB+CTB', 'NWFB',
    #            'XB'],
    #     'FERRY': ['FERRY'],
    #     'GMB': ['GMB'],
    #     'TRAM': ['PTRAM', 'TRAM'],
    #     'MTR': ['MTR'],
    #     'LR': ['LR']}
    # color_by_mode = {'FB': 'r',
    #                  'GMB': 'g',
    #                  'FERRY': 'k',
    #                  'TRAM': 'y',
    #                  'MTR': 'b',
    #                  'LR': 'c'}
    # color_by_agency = {agency: color_by_mode[mode] for mode, agencies in agency_list.items() for agency in agencies}
    # color_by_agency['FB'] = 'r'
    # print(color_by_agency)
    # mptn.plot(save='True', save_path='MPTN_network_visualization.png', color_by_label=color_by_agency)

    # mptn.network.compute_node_capacity()
    # # print(mptn.network.functional_stop_list())
    # rr = mptn.network.compute_relocation_rate(disrupted_node_list=['MTR-AUS'], max_dst=750,
    #                                           path_to_save_neighbor_dict='mptn_analyze_relocation_results/neighbor',
    #                                           read_previous_neighbor_dict='mptn_analyze_relocation_results/neighbor')
    # for k, v in rr.items():
    #     print(k, v)
