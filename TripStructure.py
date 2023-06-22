class Trip:
    def __init__(self, trip_id, route_id,
                 service_id=None, edge_list=None,
                 departure_time=None):
        self.trip_id = trip_id
        self.route_id = route_id
        self.service_id = service_id
        self.departure_time = departure_time
        if edge_list is None:
            self.edges = []
        else:
            self.edges = edge_list

    def show(self):
        print(f"{'trip_id':<20}{self.trip_id}")
        print(f"{'route_id':<20}{self.route_id}")
        print(f"{'service_id':<20}{self.service_id}")
        print(f"{'edges':<20}{self.edges}")

    def set_edge_list(self, new_edge_list=None):
        if new_edge_list is None:
            self.edges = []
        else:
            self.edges = new_edge_list

    def edge_list(self):
        return self.edges

    def is_empty(self):
        return not bool(self.edges)

    def path(self):
        if not self.edges:
            raise ValueError('empty trip edge list', self.trip_id, self.edges)
        caution = False
        for i in range(len(self.edges) - 1):
            if self.edges[i][-1] != self.edges[i+1][0]:
                caution = True
                print('caution for trip.path: disconnected path -> ', self.edges)
        return [n[0] for n in self.edges] + [self.edges[-1][-1]]
