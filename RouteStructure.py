from TripStructure import Trip
import toolbox


class Route:
    def __init__(self, route_id, agency_id=None,
                 short_name=None, long_name=None,
                 route_type=None, url=None,
                 vehicle_capacity=0):
        self.route_id = route_id
        self.trips = {}  # {trip_id:trip_class}
        # parameters
        self.agency_id = agency_id
        self.route_short_name = short_name
        self.route_long_name = long_name
        self.route_type = route_type
        self.route_url = url
        self.vehicle_capacity = vehicle_capacity

    def trip_list(self):
        return list(self.trips.keys())

    def show(self):
        print(f"{'route_id':<20}{self.route_id}")
        print(f"{'trip_list':<20}{self.trip_list()}")
        print(f"{'agency_id':<20}{self.agency_id}")
        print(f"{'route_short_name':<20}{self.route_short_name}")
        print(f"{'route_long_name':<20}{self.route_long_name}")
        print(f"{'route_type':<20}{self.route_type}")
        print(f"{'route_url':<20}{self.route_url}")
        print(f"{'capacity':<20}{self.vehicle_capacity}")

    def add_trip(self, trip_id, service_id=None, edge_list=None):
        self.trips[trip_id] = Trip(trip_id=trip_id,
                                   route_id=self.route_id,
                                   service_id=service_id,
                                   edge_list=edge_list)

    def add_multiple_trips_by_frequency(self, trip_id_prefix, frequency=1,
                                        service_id=None, edge_list=None):
        for i in range(int(frequency)):
            trip_id = trip_id_prefix + '-' + str(i)
            self.trips[trip_id] = Trip(trip_id=trip_id,
                                       route_id=self.route_id,
                                       service_id=service_id,
                                       edge_list=edge_list)

    def remove_trip(self, trip_id):
        return self.trips.pop(trip_id)

    def set_edge_list_for_trip(self, trip_id, new_edge_list):
        self.trips[trip_id].set_edge_list(new_edge_list)

    def remove_stops_from(self, list_of_stop_ids, level):  #
        return_list = []
        change = False
        for trip in self.trips.values():
            if level == 'strong':
                old_edge_list = trip.edge_list()
                new_edge_list = [(x, y) for x, y in trip.edge_list() if
                                 x not in list_of_stop_ids and y not in list_of_stop_ids]
                if new_edge_list != trip.edge_list():
                    change = True
                    trip.set_edge_list(new_edge_list)
                    divided_edge_list = toolbox.divide_sequential_edge_list(new_edge_list)
                    if divided_edge_list:
                        return_list.append(divided_edge_list)
            elif level == 'weak':
                stops_set = set([stop for edge in trip.edge_list() for stop in edge])
                if set(list_of_stop_ids).intersection(stops_set):
                    change = True
                    trip.set_edge_list([])
            else:
                raise ValueError(f'{level}, sustained level / level must be strong or weak')
        return return_list, change

    def remove_edges_from(self, list_of_edges, level):
        return_list = []
        for trip in self.trips.values():
            if level == 'strong':
                old_edge_list = trip.edge_list()
                new_edge_list = [item for item in trip.edge_list() if item not in list_of_edges]
                if new_edge_list != trip.edge_list():
                    trip.set_edge_list(new_edge_list)
                    divided_edge_list = toolbox.divide_sequential_edge_list(new_edge_list)
                    if divided_edge_list:
                        return_list.append(divided_edge_list)
                    # return_list.append(_compare_edge_list(old=old_edge_list, new=new_edge_list))
            else:
                if set(list_of_edges).intersection(set(trip.edge_list())):
                    trip.set_edge_list([])
        return return_list

    def is_empty(self):
        return not bool(self.trips)

    def remove_empty_trips(self):
        empty_trips = []
        for trip in self.trips.values():
            if trip.is_empty():
                empty_trips.append(trip.trip_id)
        for trip_id in empty_trips:
            self.remove_trip(trip_id)
        return empty_trips
        # print('empty trips have been removed from the network')

    def total_capacity(self):
        cap = 0
        for trip in self.trips.values():
            if not trip.is_empty():
                cap += self.vehicle_capacity
        return cap

    def stop_set(self):
        route_stop_list = []
        for trip in self.trips.values():
            route_stop_list.extend([stop for edge in trip.edge_list() for stop in edge])
        return set(route_stop_list)

    def edge_set(self):
        route_edge_list = []
        for trip in self.trips.values():
            route_edge_list.extend([edge for edge in trip.edge_list()])
        return set(route_edge_list)


def _compare_edge_list(old, new):
    fractures = []
    fracture = []
    if not set(new).issubset(old):
        raise ValueError('new edge_set is not a subset of old edge_set')
    k = 0
    for seq, edge in enumerate(new):
        while edge != old[seq + k]:
            k += 1
            if fracture:
                fractures.append(fracture)
                fracture = []
        fracture.append(edge)
    if fracture:
        fractures.append(fracture)
        fracture = []
    return fractures







