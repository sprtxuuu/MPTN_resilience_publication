class Stop:
    def __init__(self, stop_id, stop_name=None,
                 stop_lat=None, stop_lon=None,
                 zone_id=None, location_type=None,
                 stop_timezone=None,
                 label=None, agency_id=None):
        self.stop_id = stop_id
        self.stop_name = stop_name
        self.stop_lat = stop_lat
        self.stop_lon = stop_lon
        self.zone_id = zone_id
        self.location_type = location_type
        self.stop_timezone = stop_timezone
        self.label = label
        self.agency_id = None

    def show(self):
        print(f"{'stop_id':<20}{self.stop_id}")
        print(f"{'stop_name':<20}{self.stop_name}")
        print(f"{'stop_lat':<20}{self.stop_lat}")
        print(f"{'stop_lon':<20}{self.stop_lon}")
        print(f"{'zone_id':<20}{self.zone_id}")
        print(f"{'location_type':<20}{self.location_type}")
        print(f"{'stop_timezone':<20}{self.stop_timezone}")
        print(f"{'label':<20}{self.label}")

    def coordinates(self):
        return self.stop_lat, self.stop_lon

