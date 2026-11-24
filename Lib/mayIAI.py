import os
import time
import random
import openrouteservice as ors
import folium

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SIM_DURATION = 60
ORDER_PROB = 0.12
ORS_API_KEY = os.environ.get("ORS_API_KEY", "YOUR_KEY_HERE")

client = ors.Client(key=ORS_API_KEY)


# ---------------------------------------------------------
# TRAVEL TIME + ROUTE CACHING
# ---------------------------------------------------------

time_cache = {}
route_cache = {}

def cache_key(a, b):
    return (round(a[0], 6), round(a[1], 6),
            round(b[0], 6), round(b[1], 6))


def ors_time(start_latlon, end_latlon):
    """Return travel time in seconds using ORS with caching."""
    key = cache_key(start_latlon, end_latlon)

    if key in time_cache:
        return time_cache[key]

    try:
        coords = [
            [start_latlon[1], start_latlon[0]],
            [end_latlon[1], end_latlon[0]]
        ]
        route = client.directions(coords)
        duration = route["routes"][0]["summary"]["duration"]
        time_cache[key] = duration
        return duration

    except Exception as e:
        print("ORS TIME ERROR:", e)
        time_cache[key] = float("inf")
        return float("inf")


def ors_route_geometry(start_latlon, end_latlon):
    """Return full ORS route polyline with caching."""
    key = cache_key(start_latlon, end_latlon)

    if key in route_cache:
        return route_cache[key]

    try:
        coords = [
            [start_latlon[1], start_latlon[0]],
            [end_latlon[1], end_latlon[0]]
        ]
        route = client.directions(coords, geometry=True)
        decoded = ors.convert.decode_polyline(route["routes"][0]["geometry"])["coordinates"]

        # convert (lon,lat) → (lat,lon)
        full_path = [(lat, lon) for lon, lat in decoded]
        route_cache[key] = full_path
        return full_path

    except Exception as e:
        print("ORS ROUTE ERROR:", e)
        fallback = [start_latlon, end_latlon]
        route_cache[key] = fallback
        return fallback


# ---------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------

class Order:
    def __init__(self, id, pickup, dropoff, ts):
        self.id = id
        self.pickup = pickup
        self.dropoff = dropoff
        self.timestamp = ts
        self.assigned = None


class Vehicle:
    def __init__(self, id, pos, cap=5):
        self.id = id
        self.position = pos
        self.capacity = cap
        self.load = 0

        self.task_queue = []
        self.current_task = None
        self.time_remaining = 0

        self.completed_paths = []


# ---------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------

class DVSim:
    def __init__(self, restaurants, households, vehicles):
        self.restaurants = restaurants
        self.households = households
        self.vehicles = vehicles

        self.time = 0
        self.order_id = 0

        self.orders = []
        self.unassigned_orders = []

    def generate_order(self):
        pickup = random.choice(self.restaurants)
        drop = random.choice(self.households)
        order = Order(self.order_id, pickup, drop, self.time)

        self.order_id += 1
        self.orders.append(order)
        self.unassigned_orders.append(order)

        print(f"[{self.time}] NEW ORDER {order.id}")

    def assign_orders(self):
        for order in list(self.unassigned_orders):
            best_v = None
            best_t = float("inf")

            for v in self.vehicles:
                if v.load >= v.capacity:
                    continue
                if v.time_remaining > 0:
                    continue

                travel = ors_time(v.position, order.pickup)
                if travel < best_t:
                    best_t = travel
                    best_v = v

            if best_v:
                print(f"[{self.time}] Order {order.id} → Vehicle {best_v.id}")

                order.assigned = best_v.id
                self.unassigned_orders.remove(order)

                best_v.task_queue.append(("pickup", order))
                best_v.task_queue.append(("dropoff", order))

            else:
                print(f"[{self.time}] NO VEHICLE AVAILABLE FOR ORDER {order.id}")

    def move_vehicles(self):
        for v in self.vehicles:

            if v.time_remaining > 0:
                v.time_remaining -= 1
                continue

            if v.current_task:
                ttype, order = v.current_task
                if ttype == "pickup":
                    v.load += 1
                    print(f"[{self.time}] Vehicle {v.id} PICKED Order {order.id}")
                else:
                    v.load -= 1
                    print(f"[{self.time}] Vehicle {v.id} DROPPED Order {order.id}")
                v.current_task = None

            if v.task_queue:
                ttype, order = v.task_queue.pop(0)
                target = order.pickup if ttype == "pickup" else order.dropoff

                travel_t = ors_time(v.position, target)
                geometry = ors_route_geometry(v.position, target)

                v.completed_paths.append(geometry)
                v.position = target
                v.current_task = (ttype, order)
                v.time_remaining = int(travel_t)

                print(f"[{self.time}] Vehicle {v.id} START {ttype} Order {order.id} "
                      f"(ETA {travel_t/60:.2f}m)")

    def step(self):
        if random.random() < ORDER_PROB:
            self.generate_order()

        self.assign_orders()
        self.move_vehicles()

    def run(self):
        print("Starting simulation...\n")
        while self.time < SIM_DURATION:
            self.step()
            time.sleep(0.05)
            self.time += 1
        print("\nSimulation finished.")


# ---------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------

if __name__ == "__main__":

    restaurants = [
        [51.481329, -0.010813],
        [51.481042, -0.008780],
        [51.479680, -0.0091387],
        [51.481822, -0.010334],
        [51.475921, -0.018602],

        # NEW RESTAURANTS (corrected)
        [51.480634, -0.020474],
        [51.482662, -0.019047],
        [51.477982, -0.026002]
    ]

    households = [
        [51.482000, -0.015000],
        [51.480500, -0.025000],
        [51.483200, -0.020800]
        # more households can be added later
    ]

    vehicles = [
        Vehicle(1, [51.477890, -0.012200]),
        Vehicle(2, [51.482155, -0.005159]),
    ]

    sim = DVSim(restaurants, households, vehicles)
    sim.run()

    print("Generating map...")

    m = folium.Map(location=restaurants[0], zoom_start=14, tiles="CartoDB Positron")

    for r in restaurants:
        folium.Marker(r, icon=folium.Icon(color="blue")).add_to(m)

    for h in households:
        folium.Marker(h, icon=folium.Icon(color="green")).add_to(m)

    for v in vehicles:
        folium.Marker(v.position, icon=folium.Icon(color="red")).add_to(m)
        for path in v.completed_paths:
            folium.PolyLine(path, color="orange", weight=3).add_to(m)

    m.save("simulation_map.html")
    print("Map saved as simulation_map.html")
