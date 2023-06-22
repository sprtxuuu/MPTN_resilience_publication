from RouteStructure import Route
from StopStructure import Stop
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import json
from scipy import stats


class Network:
    def __init__(self, network_name):
        self.network_name = network_name
        self.routes = {}  # system structure
        self.stop_repository = {}  # NOTE: standalone stop repository
        self.edge_capacity = {}
        self.node_capacity = {}
        # self.node_coordinates = {}

    def show(self):
        # print(f"|V|={self.G.number_of_nodes()}, |E|={self.G.number_of_edges()} (note to update)")
        # print(f"{'number of routes':<20}{len(self.route_list())}")
        # print(f"{'route_list':<20}{self.route_list()}")
        # print(f"{'number of stops':<20}{len(self.stop_repository_list())}")
        # print(f"{'stop_list':<20}{self.stop_repository_list()}")
        sort_by_agency = {}
        for route_id in self.route_list():
            agency = self.routes[route_id].agency_id
            if agency in sort_by_agency.keys():
                sort_by_agency[agency]["count"] += 1
                sort_by_agency[agency]["num_of_trips"] += len(self.routes[route_id].trip_list())
                sort_by_agency[agency]["max_cap"] += self.routes[route_id].total_capacity()
                sort_by_agency[agency]["stops"] = sort_by_agency[agency]["stops"].union(
                    self.routes[route_id].stop_set())
                sort_by_agency[agency]["edges"] = sort_by_agency[agency]["edges"].union(
                    self.routes[route_id].edge_set())
            else:
                sort_by_agency[agency] = {"count": 1,
                                          "num_of_trips": len(self.routes[route_id].trip_list()),
                                          "max_cap": self.routes[route_id].total_capacity(),
                                          "stops": self.routes[route_id].stop_set(),
                                          "edges": self.routes[route_id].edge_set(),
                                          "veh_cap": self.routes[route_id].vehicle_capacity}
        if sort_by_agency:
            print('-' * 200)
            print(
                f"{'Agency':<10}{'number_of_routes':>20}{'num_of_trips':>20}{'veh_cap':>20}{'max_cap':>20}{'num_of_stops':>20}{'num_of_edges':>20}")
            for k, v in sort_by_agency.items():
                count = v['count']
                num_of_trips = v['num_of_trips']
                max_cap = v['max_cap']
                num_of_stops = len(v['stops'])
                num_of_edges = len(v['edges'])
                veh_cap = v['veh_cap']
                print(
                    f"{k:<10}{count:>20,}{num_of_trips:>20,}{veh_cap:>20,}{max_cap:>20,}{num_of_stops:>20,}{num_of_edges:>20,}")
        # print('#' * 50)
        # fbus =['CTB', 'KMB+CTB', 'KMB', 'NLB', 'PI', 'LWB', 'DB', 'LRTFeeder', 'KMB+NWFB', 'LWB+CTB', 'NWFB',
        #        'XB']
        # print(f'FBus total={np.sum([sort_by_agency[agcy]["max_cap"] for agcy in fbus])}')

    def add_route(self, route_id, agency_id=None,
                  route_short_name=None, route_long_name=None,
                  route_type=None, route_url=None, vehicle_capacity=0):
        # if route_id in self.routes.keys():
        #     print(f'Warning: route {route_id} already exist, will override')
        self.routes[route_id] = Route(route_id=route_id,
                                      agency_id=agency_id,
                                      short_name=route_short_name,
                                      long_name=route_long_name,
                                      route_type=route_type,
                                      url=route_url,
                                      vehicle_capacity=vehicle_capacity)

    def add_stop(self, stop_id, stop_name=None,
                 stop_lat=None, stop_lon=None,
                 zone_id=None, location_type=None,
                 stop_timezone=None,
                 label=None):
        self.stop_repository[stop_id] = Stop(stop_id=stop_id, stop_name=stop_name,
                                             stop_lat=stop_lat, stop_lon=stop_lon,
                                             zone_id=zone_id, location_type=location_type,
                                             stop_timezone=stop_timezone,
                                             label=label)

    def route_list(self):
        return list(self.routes.keys())

    def stop_repository_list(self):  # stop repository
        return list(self.stop_repository.keys())

    def stop_set(self):
        stop_set = set()
        for route in self.routes.values():
            stop_set = stop_set.union(route.stop_set())
        return stop_set

    def functional_stop_list(self):
        return list(self.stop_set())

    def edge_set(self):
        edge_set = set()
        for route in self.routes.values():
            edge_set = edge_set.union(route.edge_set())
        return edge_set

    def look_up_trip_id(self, trip_id):
        # assume that a trip belongs to only one route
        for route_id, route in self.routes.items():
            if trip_id in route.trip_list():
                return route_id

    def remove_routes_by_agency(self, agency_id):
        for route_id in self.route_list():
            if self.routes[route_id].agency_id == agency_id:
                self.routes.pop(route_id)

    def remove_route(self, route_id):
        return self.routes.pop(route_id)

    def print_all_routes_detail(self):
        for route in self.routes.values():
            route.show()

    def print_all_trips_detail(self):
        for route in self.routes.values():
            for trip in route.trips.values():
                trip.show()

    def print_empty_routes(self, remove_empty_routes=False):
        print('#' * 50)
        print('empty routes found:')
        empty_routes = []
        for route in self.routes.values():
            if route.is_empty():
                print(route.route_id, route.agency_id)
                empty_routes.append(route.route_id)
        if remove_empty_routes:
            for route_id in empty_routes:
                self.remove_route(route_id)
            print('empty routes have been removed from the network')
        return empty_routes

    def remove_empty_routes(self):
        empty_routes = []
        for route in self.routes.values():
            if route.is_empty():
                empty_routes.append(route.route_id)
        for route_id in empty_routes:
            self.remove_route(route_id)
        return empty_routes
        # print('empty routes have been removed from the network')

    def remove_stops_from(self, list_of_stop_ids, level):
        return_list = []
        changes = []
        for route in self.routes.values():
            single_return, change = route.remove_stops_from(list_of_stop_ids, level)
            if single_return:
                return_list.extend(single_return)
            changes.append(change)
        return return_list, changes

    def remove_edges_from(self, list_of_edges, level):
        return_list = []
        for route in self.routes.values():
            single_return = route.remove_edges_from(list_of_edges, level)
            if single_return:
                return_list.extend(single_return)
        return return_list

    def compute_edge_capacity(self):
        self.edge_capacity = {}
        for route in self.routes.values():
            for trip in route.trips.values():
                for edge in trip.edges:
                    if edge not in self.edge_capacity.keys():
                        self.edge_capacity[edge] = route.vehicle_capacity
                    else:
                        self.edge_capacity[edge] += route.vehicle_capacity

    def compute_total_capacity(self):
        total_cap = 0
        for route in self.routes.values():
            total_cap += route.total_capacity()
        return total_cap

    def load_from_gtfs_dataset(self, folder_path, weight=False, walking_dst=0, peak_hour=False):
        # vehicle capacity (person)
        # #vehicle capacity: Upper Deck Seating: 63 Lower Deck Seating: 35 Lower Deck Standees: 48
        # # 63+35+48 / double decker
        # # http://www.kmb.hk/en/news/press/archives/news201408222062.html
        # # 19 / minibus
        # # 115 / tram
        # # https://en.wikipedia.org/wiki/Hong_Kong_Tramways
        # # 300, 388, 410, 170-200, 180 / ferry
        # # http://www.hkkf.com.hk/index.php?op=show&page=fleet&style=en
        # # 3750 / train
        # # https://www.mtr.com.hk/en/corporate/operations/detail_worldclass.html
        vc_by_mode = {'FB': 146,
                      'GMB': 19,
                      'FERRY': 150,
                      'TRAM': 115}
        agency_list = {
            'FB': ['CTB', 'KMB+CTB', 'KMB', 'NLB', 'PI', 'LWB', 'DB', 'LRTFeeder', 'KMB+NWFB', 'LWB+CTB', 'NWFB',
                   'XB'],
            'FERRY': ['FERRY'],
            'GMB': ['GMB'],
            'TRAM': ['PTRAM', 'TRAM']}
        mode_of_agency = {agency: mode for mode, agencies in agency_list.items() for agency in agencies}
        vc_by_agency = {agency: vc_by_mode[mode] for mode, agencies in agency_list.items() for agency in agencies}

        # 1. read gtfs files
        agency = pd.read_csv(folder_path + '/agency.txt', header=0)
        calendar = pd.read_csv(folder_path + '/calendar.txt', header=0)
        calendar_dates = pd.read_csv(folder_path + '/calendar_dates.txt', header=0)
        fare_attributes = pd.read_csv(folder_path + '/fare_attributes.txt', header=0)
        fare_rules = pd.read_csv(folder_path + '/fare_rules.txt', header=0)
        frequencies = pd.read_csv(folder_path + '/frequencies.txt', header=0)
        routes = pd.read_csv(folder_path + '/routes.txt', header=0, index_col=0)
        stop_times = pd.read_csv(folder_path + '/stop_times.txt', header=0)
        stops = pd.read_csv(folder_path + '/stops.txt', header=0, index_col=0)
        trips = pd.read_csv(folder_path + '/trips.txt', header=0)
        gtfs = [agency, calendar, calendar_dates, fare_attributes, fare_rules, frequencies, routes, stop_times, stops,
                trips]
        # print_dataframe(agency,'\n')

        # 2. load stops
        node_param = stops.to_dict(orient='index')
        for stop_id, param in node_param.items():
            stop_id = str(stop_id)
            self.stop_repository[stop_id] = Stop(stop_id=stop_id, stop_name=param['stop_name'],
                                                 stop_lat=float(param['stop_lat']), stop_lon=float(param['stop_lon']),
                                                 zone_id=param['zone_id'], location_type=param['location_type'],
                                                 stop_timezone=param['stop_timezone'])
            # self.node_coordinates[stop_id] = (float(param['stop_lat']), float(param['stop_lon']))  # (lat, lon)

        # load routes
        route_list = routes.index.tolist()
        route_param = routes.to_dict(orient='dict')
        for route_id in route_list:
            self.routes[str(route_id)] = Route(route_id=str(route_id),
                                               agency_id=route_param['agency_id'][route_id],
                                               short_name=route_param['route_short_name'][route_id],
                                               long_name=route_param['route_long_name'][route_id],
                                               route_type=route_param['route_type'][route_id],
                                               url=route_param['route_url'][route_id],
                                               vehicle_capacity=vc_by_agency[route_param['agency_id'][route_id]])

        # 3. load trips
        trip_param = trips.set_index('trip_id').to_dict()
        trip_grp = trips.groupby(by='route_id')
        route_trip = defaultdict(list)
        for route_id, item in trip_grp:
            trip_list = trip_grp.get_group(route_id)['trip_id'].tolist()
            service_list = trip_grp.get_group(route_id)['service_id'].tolist()
            for i, trip_id in enumerate(trip_list):
                self.routes[str(route_id)].add_trip(trip_id=trip_id, service_id=service_list[i])
                route_trip[str(route_id)].append(str(trip_id))

        from xc_resilience_live import revert_dict_of_list
        trip_route = revert_dict_of_list(dict(route_trip))

        # 4. load stop sequence (based on trips) and create unweighted network
        # self.route['1001'].trip['1001-1-287-0535'].show()
        stop_times_grp = stop_times[['trip_id', 'stop_id', 'stop_sequence']].groupby(by='trip_id')
        for trip_id, item in stop_times_grp:
            time_index = int(str.split(trip_id, '-')[-1])
            if 700 < time_index < 1000 or 1700 < time_index < 2000:
                in_peak = 1
            else:
                in_peak = -1
            if peak_hour * in_peak < 0:  # peak_hour and in_peak do not match
                continue
            else:  # peak_hour and in_peak do match or peak_hour=False
                stop_seq_table = stop_times_grp.get_group(trip_id).sort_values(by='stop_sequence', ascending=True)
                stop_seq = stop_seq_table['stop_id'].tolist()
                for stop_id in stop_seq:
                    agency_id = self.routes[trip_route[str(trip_id)][0]].agency_id
                    self.stop_repository[str(stop_id)].label = mode_of_agency[agency_id]
                edge_list = [(str(stop_seq[i]), str(stop_seq[i + 1])) for i in range(len(stop_seq) - 1)]
                route_id = self.look_up_trip_id(trip_id)
                self.routes[route_id].set_edge_list_for_trip(trip_id, edge_list)
        for route in self.routes.values():
            route.remove_empty_trips()
        self.remove_empty_routes()

    def generate_intermodal_edges(self, dst_limit,
                                  path_to_save_neighbor_dict=False,
                                  read_previous_neighbor_dict=False):
        # use stop.label for modes
        intermodal_edge_list = []
        if read_previous_neighbor_dict:
            neighbor = load_pet(path_to_save_neighbor_dict)
            print(f'read neighborhood dict from {path_to_save_neighbor_dict}...')
        else:
            neighbor = self.generate_neighborhood_dict(path_to_save=path_to_save_neighbor_dict)
        functional_stop_list = self.functional_stop_list()
        for u in functional_stop_list:
            if u in neighbor.keys():
                label_u = self.stop_repository[u].label
                for nb in neighbor[u]:
                    v, dst = nb
                    if v in functional_stop_list and dst < dst_limit:
                        label_v = self.stop_repository[v].label
                        if label_u != label_v:
                            intermodal_edge_list.append((u, v))
        print('number of intermodal edges =', len(intermodal_edge_list))
        return intermodal_edge_list

    def generate_neighborhood_dict(self, dst_max=1600, path_to_save=None):
        """{node:[(neighbor, distance),...]}"""
        from xc_resilience_live import haversine
        neighbor = {stop: [] for stop in self.functional_stop_list()}
        print('computing neighborhood dict...')
        for cb in combinations(list(self.functional_stop_list()), 2):
            lat1, lon1 = self.stop_repository[cb[0]].coordinates()
            lat2, lon2 = self.stop_repository[cb[1]].coordinates()
            dst = haversine(lat1, lon1, lat2, lon2)
            if dst < dst_max:
                neighbor[cb[0]].append((cb[1], dst))
                neighbor[cb[1]].append((cb[0], dst))
        if path_to_save:
            print('neighborhood dict generated...')
            save_pet(dict(neighbor), path_to_save)
        return dict(neighbor)

    def compute_node_capacity(self):
        self.node_capacity = {}
        for route in self.routes.values():
            for trip in route.trips.values():
                for edge in trip.edges:
                    for node in edge:
                        if node not in self.node_capacity.keys():
                            self.node_capacity[node] = 0.5 * route.vehicle_capacity / len(trip.edges)
                        else:
                            self.node_capacity[node] += 0.5 * route.vehicle_capacity / len(trip.edges)
        return self.node_capacity

    def compute_relocation_rate(self, disrupted_node_list, max_dst,
                                read_previous_neighbor_dict=False,
                                single_disruption_analysis=False,
                                minimum_distance_tolerance=10.0):
        def df(d, max_d=max_dst):
            will = 1 - d / max_d
            if 0 <= will <= 1:
                return will
            else:
                return 0

        relocation_rate = {stop: [] for stop in disrupted_node_list}
        if read_previous_neighbor_dict:
            neighbor = load_pet(read_previous_neighbor_dict)
        else:
            print(f'fail to read neighborhood dict, generating new one to {read_previous_neighbor_dict}...')
            neighbor = self.generate_neighborhood_dict(path_to_save=read_previous_neighbor_dict)

        for v_d in disrupted_node_list:
            if not single_disruption_analysis:
                valid_neighbors = [[nb, float(dst)] for nb, dst in neighbor[v_d] if
                                   nb not in disrupted_node_list and dst < max_dst]
            else:
                valid_neighbors = [[nb, float(dst)] for nb, dst in neighbor[v_d] if dst < max_dst]
            if valid_neighbors:
                nb_array = np.asarray(valid_neighbors)[:, 0]
                dst_array = np.asarray(valid_neighbors)[:, 1].astype(float)
                dst_array = np.array(
                    [value if value > minimum_distance_tolerance else minimum_distance_tolerance for value in
                     dst_array])
                closeness = dst_array ** (-1) / np.sum(dst_array ** (-1))
                dst_factor = np.asarray([df(v) for v in dst_array])
                rates = np.vstack((nb_array, dst_array, closeness * dst_factor)).T.tolist()
                rates = sorted(rates, key=lambda x: x[1])
                relocation_rate[v_d] = rates
        return relocation_rate

    def agency_list(self):
        from xc_resilience_live import remove_duplicate
        agencies = [self.routes[route_id].agency_id for route_id in self.route_list()]
        return remove_duplicate(agencies)

    def add_agency_id_to_stops(self):
        for route in self.routes.values():
            agency = route.agency_id
            for stop in list(route.stop_set()):
                self.stop_repository[stop].agency_id = agency



def save_pet(pet, filename='temporary file'):
    with open(filename, 'w') as f:
        f.write(json.dumps(str(pet)))


def load_pet(filename):
    with open(filename) as f:
        pet = json.loads(f.read())
    return eval(pet)
