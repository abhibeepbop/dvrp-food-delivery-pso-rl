import random
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

GRID_SIZE = 100
SIM_DURATION = 120            # seconds
ORDER_PROB_PER_HH = 0.03
VEHICLE_SPEED = 1.0           # grid cells per sec
NUM_VEHICLES = 3
LATE_DEADLINE = 60            # seconds

DISPATCH_POLICY = "nearest"   # "nearest" or "random"
random.seed(42)


# ---------------------------------------------------------
# FIXED RESTAURANTS (10)
# ---------------------------------------------------------

restaurants_coords = [
    (20, 20), (22, 21), (19, 23),
    (50, 50), (52, 49), (48, 51),
    (75, 75), (78, 76), (74, 73),
    (10, 90)
]

# ---------------------------------------------------------
# FIXED HOUSEHOLDS (10)
# ---------------------------------------------------------

households_coords = [
    (10, 70), (10, 72), (10, 75),
    (50, 60), (60, 50),
    (80, 20), (82, 22), (78, 18), (90, 90), (10, 2)
]


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def travel_time(a, b):
    dist = manhattan(a, b)
    return int(max(1, dist / VEHICLE_SPEED)) if dist > 0 else 0


# ---------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------

class Restaurant:
    def __init__(self, rid, pos):
        self.id = rid
        self.pos = pos


class Household:
    def __init__(self, hid, pos):
        self.id = hid
        self.pos = pos
        self.busy = False
        self.current_order = None


class Order:
    def __init__(self, oid, restaurant, household, created_at):
        self.id = oid
        self.restaurant = restaurant
        self.household = household
        self.created_at = created_at

        self.picked_at = None
        self.delivered_at = None

        self.assigned_vehicle = None
        self.delivered = False


class Vehicle:
    def __init__(self, vid, pos):
        self.id = vid
        self.pos = pos

        self.state = "idle"      # idle, to_pickup, to_dropoff
        self.time_remaining = 0
        self.order = None

        # utilisation tracking
        self.idle_time = 0
        self.pickup_time = 0
        self.dropoff_time = 0

        # distance tracking
        self.distance_travelled = 0


# ---------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------

def plot_points(restaurants, households):
    plt.figure(figsize=(7, 7))

    plt.scatter([r.pos[0] for r in restaurants],
                [r.pos[1] for r in restaurants],
                color='orange', marker='X', s=80, label="Restaurants")

    plt.scatter([h.pos[0] for h in households],
                [h.pos[1] for h in households],
                color='blue', s=60, label="Households")

    plt.title("100×100 Grid — Restaurants & Households")
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# SIMULATION CLASS
# ---------------------------------------------------------

class DVSim:
    def __init__(self, restaurants, households, vehicles):
        self.restaurants = restaurants
        self.households = households
        self.vehicles = vehicles

        self.time = 0
        self.next_order_id = 0

        self.orders = []
        self.pending_orders = []
        self.completed_orders = []

    # --------------------- ORDER GENERATION ---------------------

    def generate_orders(self):
        for hh in self.households:
            if hh.busy:
                continue
            if random.random() < ORDER_PROB_PER_HH:
                r = random.choice(self.restaurants)
                order = Order(self.next_order_id, r, hh, self.time)
                self.next_order_id += 1

                hh.busy = True
                hh.current_order = order

                self.orders.append(order)
                self.pending_orders.append(order)

                print(f"[{self.time}] NEW ORDER {order.id}: HH {hh.id} → R {r.id}")

    # --------------------- BASELINE DISPATCH POLICIES ---------------------

    def assign_orders(self):
        for order in list(self.pending_orders):

            if DISPATCH_POLICY == "random":
                idle = [v for v in self.vehicles if v.state == "idle"]
                if not idle:
                    continue
                chosen = random.choice(idle)

            elif DISPATCH_POLICY == "nearest":
                best_v = None
                best_t = float("inf")
                for v in self.vehicles:
                    if v.state != "idle":
                        continue
                    t = travel_time(v.pos, order.restaurant.pos)
                    if t < best_t:
                        best_t = t
                        best_v = v
                chosen = best_v

            else:
                raise ValueError("Invalid dispatch policy")

            if chosen:
                chosen.state = "to_pickup"
                chosen.order = order
                chosen.time_remaining = travel_time(chosen.pos, order.restaurant.pos)
                chosen.distance_travelled += manhattan(chosen.pos, order.restaurant.pos)

                order.assigned_vehicle = chosen
                self.pending_orders.remove(order)

                print(f"[{self.time}] ORDER {order.id} → VEH {chosen.id}")

    # --------------------- VEHICLE MOVEMENT ---------------------

    def move_vehicles(self):
        for v in self.vehicles:

            # utilisation tracking
            if v.state == "idle":
                v.idle_time += 1
            elif v.state == "to_pickup":
                v.pickup_time += 1
            elif v.state == "to_dropoff":
                v.dropoff_time += 1

            if v.time_remaining > 0:
                v.time_remaining -= 1
                continue

            if v.state == "idle":
                continue

            order = v.order

            # Arrived at restaurant
            if v.state == "to_pickup":
                v.pos = order.restaurant.pos
                v.state = "to_dropoff"

                order.picked_at = self.time
                t = travel_time(v.pos, order.household.pos)
                v.time_remaining = t
                v.distance_travelled += manhattan(v.pos, order.household.pos)

                print(f"[{self.time}] VEH {v.id} PICKED ORDER {order.id}")

            # Delivered order
            elif v.state == "to_dropoff":
                v.pos = order.household.pos
                v.state = "idle"

                order.delivered = True
                order.delivered_at = self.time
                self.completed_orders.append(order)

                # free household
                order.household.busy = False
                order.household.current_order = None

                print(f"[{self.time}] VEH {v.id} DROPPED ORDER {order.id}")

                v.order = None

    # --------------------- RUN SIMULATION ---------------------

    def run(self, steps=SIM_DURATION):
        print("Starting simulation...\n")
        for t in range(steps):
            self.time = t
            self.generate_orders()
            self.assign_orders()
            self.move_vehicles()

        print("\nSimulation complete.")
        self.print_metrics()

    # --------------------- METRICS ---------------------

    def print_metrics(self):
        total_orders = len(self.orders)
        completed = len(self.completed_orders)

        print("\n===== RESULTS =====")
        print(f"Orders Created: {total_orders}")
        print(f"Orders Completed: {completed}")

        if completed > 0:
            adt = sum(o.delivered_at - o.created_at for o in self.completed_orders) / completed
            apt = sum(o.picked_at - o.created_at for o in self.completed_orders) / completed
            late = sum(1 for o in self.completed_orders if (o.delivered_at - o.created_at) > LATE_DEADLINE)

            print(f"Average Delivery Time: {adt:.2f} sec")
            print(f"Average Pickup Time: {apt:.2f} sec")
            print(f"Late Deliveries: {late}")

        total_distance = sum(v.distance_travelled for v in self.vehicles)
        print(f"Total Distance Travelled: {total_distance}")

        for v in self.vehicles:
            total = v.idle_time + v.pickup_time + v.dropoff_time
            print(f"Vehicle {v.id} Utilisation => "
                  f"Idle: {v.idle_time/total:.2f}, "
                  f"Pickup: {v.pickup_time/total:.2f}, "
                  f"Dropoff: {v.dropoff_time/total:.2f}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    restaurants = [Restaurant(i, pos) for i, pos in enumerate(restaurants_coords)]
    households = [Household(i, pos) for i, pos in enumerate(households_coords)]

    print("Plotting static map...")
    plot_points(restaurants, households)

    possible_starts = restaurants_coords + households_coords
    vehicles = [Vehicle(i, random.choice(possible_starts)) for i in range(NUM_VEHICLES)]

    sim = DVSim(restaurants, households, vehicles)
    sim.run()
