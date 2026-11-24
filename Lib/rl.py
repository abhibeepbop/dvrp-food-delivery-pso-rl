# THIS IS THE FILE USED FOR THE RL ACTUALLY USED IN THE PAPER


import random
import numpy as np
import matplotlib.pyplot as plt

# CONFIGURATION HERE

GRID_SIZE = 10
SIM_DURATION = 300

VEHICLE_SPEED = 1.0
VEHICLE_CAPACITY = 2
NUM_VEHICLES = 5

# dynamic orders
BASE_ORDER_RATE = 22          # orders per episode (dynamic)

# RL EPISODES
RL_EPISODES = 30000

# epsilon decay
EPS_START = 0.40
EPS_END   = 0.001
EPS_DECAY = 0.995

# learning rate
ALPHA = 0.12

# reward coefficients
DISTANCE_COST = 0.08
LATE_PENALTY = 50
LATE_DEADLINE = 60

random.seed(42)

# Locations of restaurants and households (these are fixed)
restaurants_coords = [
    (20, 20), (22, 21), (19, 23),
    (50, 50), (52, 49), (48, 51),
    (75, 75), (78, 76), (74, 73),
    (10, 90)
]

households_coords = [
    (10, 70), (10, 72), (10, 75),
    (50, 60), (60, 50),
    (80, 20), (82, 22), (78, 18),
    (90, 90), (10, 2)
]

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def travel_time(a, b):
    return max(1, int(manhattan(a, b) / VEHICLE_SPEED))

# Data Structures

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
    def __init__(self, oid, r, h, created):
        self.id = oid
        self.restaurant = r
        self.household = h
        self.created_at = created
        self.picked_at = None
        self.delivered_at = None
        self.assigned_vehicle = None
        self.delivered = False

class Vehicle:
    def __init__(self, vid, pos):
        self.id = vid
        self.pos = pos
        self.state = "idle"        # idle, pickup, dropoff
        self.time_remaining = 0
        self.orders = []           # task buffer
        self.current_task = None
        self.load = 0
        self.distance = 0
        self.idle_time = 0
        self.pickup_time = 0
        self.dropoff_time = 0
# ---------------------------------------------------------
# ASSIGNMENT RL AGENT (PER-VEHICLE)
# ---------------------------------------------------------

class AssignmentAgent:

    def __init__(self):
        # buckets
        self.dist_buckets = 4
        self.load_buckets = 3
        self.idle_buckets = 2
        self.pending_buckets = 3

        # total states per vehicle
        self.states = (
            self.dist_buckets *
            self.load_buckets *
            self.idle_buckets *
            self.pending_buckets
        )

        # actions: 0=reject, 1=accept
        self.actions = 2

        # Q-tables: one per vehicle
        self.Q = np.zeros((NUM_VEHICLES, self.states, self.actions))

        # memory for updates
        self.memory = {}   # order_id → (vehicle_id, state, action)

    # -------------------------------
    # bucket helpers
    # -------------------------------

    def bucket_dist(self, d):
        if d <= 10: return 0
        if d <= 25: return 1
        if d <= 50: return 2
        return 3

    def bucket_load(self, load):
        return min(load, 2)

    def bucket_idle(self, idle_flag):
        return 1 if idle_flag else 0

    def bucket_pending(self, p):
        if p <= 2: return 0
        if p <= 5: return 1
        return 2

    # -------------------------------
    # build state index
    # -------------------------------

    def build_state(self, vehicle, order, pending_count):

        dist = manhattan(vehicle.pos, order.restaurant.pos)
        d = self.bucket_dist(dist)

        load_b = self.bucket_load(vehicle.load)

        idle_flag = 1 if vehicle.state == "idle" else 0
        idle_b = self.bucket_idle(idle_flag)

        pend_b = self.bucket_pending(pending_count)

        idx = (
            d +
            self.dist_buckets * (
                load_b +
                self.load_buckets * (
                    idle_b +
                    self.idle_buckets * pend_b
                )
            )
        )

        return idx

    # -------------------------------
    # action selection (ε-greedy)
    # -------------------------------

    def select_action(self, vehicle_id, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.actions)
        return np.argmax(self.Q[vehicle_id, state])

    # -------------------------------
    # remember assignment
    # -------------------------------

    def remember(self, order_id, vehicle_id, state, action):
        self.memory[order_id] = (vehicle_id, state, action)

    # -------------------------------
    # reward after delivery
    # -------------------------------

    def update(self, order, delivery_time, vehicle_distance):
        if order.id not in self.memory:
            return

        v_id, st, act = self.memory.pop(order.id)

        late_flag = 1 if delivery_time > LATE_DEADLINE else 0

        reward = -(
                (delivery_time / 60) +
                (vehicle_distance / 300) +
                3 * late_flag
        )

        old = self.Q[v_id, st, act]
        self.Q[v_id, st, act] = old + ALPHA * (reward - old)
# ---------------------------------------------------------
# ROUTING RL AGENT
# ---------------------------------------------------------

class RoutingAgent:

    def __init__(self):
        # buckets for state
        self.task_buckets = 3        # 0,1,2+
        self.dist_buckets = 4
        self.wait_buckets = 4

        # total states
        self.states = (
            self.task_buckets *
            self.dist_buckets *
            self.wait_buckets
        )

        # actions:
        # 0 = pickup nearest
        # 1 = dropoff nearest
        # 2 = overall nearest
        # 3 = FIFO (oldest order)
        # 4 = best distance/wait score
        self.actions = 5

        # Q-table per vehicle
        self.Q = np.zeros((NUM_VEHICLES, self.states, self.actions))

        # memory for updates
        self.memory = {}   # (veh_id, order_id) → (state, action)

    # -------------------------------
    # bucket helpers
    # -------------------------------

    def bucket_tasks(self, t):
        if t == 0: return 0
        if t == 1: return 1
        return 2

    def bucket_dist(self, d):
        if d <= 10: return 0
        if d <= 25: return 1
        if d <= 50: return 2
        return 3

    def bucket_wait(self, w):
        if w <= 20: return 0
        if w <= 40: return 1
        if w <= 80: return 2
        return 3

    # -------------------------------
    # build routing state
    # -------------------------------

    def build_state(self, vehicle, current_time):

        t = len(vehicle.orders)
        t_b = self.bucket_tasks(t)

        if t > 0:
            d0 = manhattan(vehicle.pos, vehicle.orders[0].restaurant.pos)
        else:
            d0 = 999

        d_b = self.bucket_dist(d0)

        if t > 0:
            max_wait = max(current_time - o.created_at for o in vehicle.orders)
        else:
            max_wait = 0

        w_b = self.bucket_wait(max_wait)

        idx = (
            t_b +
            self.task_buckets * (
                d_b +
                self.dist_buckets * w_b
            )
        )

        return idx

    # -------------------------------
    # choose routing action
    # -------------------------------

    def select_action(self, vehicle_id, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.actions)
        return np.argmax(self.Q[vehicle_id, state])

    # -------------------------------
    # remember routing action
    # -------------------------------

    def remember(self, vehicle_id, order_id, state, action):
        self.memory[(vehicle_id, order_id)] = (state, action)

    # -------------------------------
    # update after delivery
    # -------------------------------

    def update(self, vehicle_id, order, delivery_time, distance_travelled):
        key = (vehicle_id, order.id)
        if key not in self.memory:
            return

        st, ac = self.memory.pop(key)

        late_flag = 1 if delivery_time > LATE_DEADLINE else 0

        reward = -(
            delivery_time +
            DISTANCE_COST * distance_travelled +
            LATE_PENALTY * late_flag
        )

        old = self.Q[vehicle_id, st, ac]
        self.Q[vehicle_id, st, ac] = old + ALPHA * (reward - old)
# ---------------------------------------------------------
# SIMULATION CLASS
# ---------------------------------------------------------

class DVSim:

    def __init__(self, restaurants, households, vehicles,
                 assign_agent=None, route_agent=None,
                 use_assign_rl=False, use_route_rl=False,
                 epsilon=0.1):

        self.restaurants = restaurants
        self.households = households
        self.vehicles = vehicles

        self.assign_agent = assign_agent
        self.route_agent = route_agent

        self.use_assign_rl = use_assign_rl
        self.use_route_rl = use_route_rl

        self.epsilon = epsilon
        self.time = 0

        self.next_order_id = 0
        self.orders = []
        self.pending = []
        self.completed = []

        # randomised order timings
        self.order_times = sorted(random.sample(range(10, SIM_DURATION), BASE_ORDER_RATE))

    # -----------------------------------------------------
    # ORDER GENERATION
    # -----------------------------------------------------

    def generate_orders(self):
        if self.time not in self.order_times:
            return

        hh = random.choice(self.households)
        if hh.busy:
            # pick another household that is not busy
            free = [x for x in self.households if not x.busy]
            if not free:
                return  # no free households, skip just this timestamp
            hh = random.choice(free)

        r = random.choice(self.restaurants)

        order = Order(self.next_order_id, r, hh, self.time)
        self.next_order_id += 1

        hh.busy = True
        hh.current_order = order

        self.orders.append(order)
        self.pending.append(order)

    # -----------------------------------------------------
    # ASSIGNMENT LOGIC
    # -----------------------------------------------------

    def assign_orders(self):

        for order in list(self.pending):

            pending_count = len(self.pending)

            # -----------------------
            # RL ENABLED
            # -----------------------
            if self.use_assign_rl and self.assign_agent is not None:

                scores = {}
                memories = {}

                for v in self.vehicles:

                    cap_left = VEHICLE_CAPACITY - v.load
                    if cap_left <= 0:
                        continue

                    st = self.assign_agent.build_state(v, order, pending_count)
                    ac = self.assign_agent.select_action(v.id, st, self.epsilon)

                    if ac == 1:
                        val = self.assign_agent.Q[v.id, st, ac]
                        scores[v.id] = val
                        memories[v.id] = (st, ac)

                if scores:
                    chosen_id = max(scores, key=scores.get)
                    chosen = self.vehicles[chosen_id]
                    (st, ac) = memories[chosen_id]
                    self.assign_agent.remember(order.id, chosen_id, st, ac)

                else:
                    available = [v for v in self.vehicles if (VEHICLE_CAPACITY - v.load) > 0]
                    chosen = min(available, key=lambda vv:
                                 manhattan(vv.pos, order.restaurant.pos))

            # -----------------------
            # BASELINE
            # -----------------------
            else:
                available = [v for v in self.vehicles if (VEHICLE_CAPACITY - v.load) > 0]
                if not available:
                    continue
                chosen = min(available,
                             key=lambda vv: manhattan(vv.pos, order.restaurant.pos))

            # -----------------------
            # APPLY ASSIGNMENT
            # -----------------------
            chosen.orders.append(order)
            chosen.load += 1
            order.assigned_vehicle = chosen
            self.pending.remove(order)

            if chosen.state == "idle" and chosen.time_remaining == 0:
                self.start_next_task(chosen)

    # -----------------------------------------------------
    # CHOOSE NEXT TASK (Routing RL)
    # -----------------------------------------------------

    def choose_next_task(self, v):

        if len(v.orders) == 0:
            return None

        if not self.use_route_rl or self.route_agent is None:
            return self.closest_task(v)

        st = self.route_agent.build_state(v, self.time)
        ac = self.route_agent.select_action(v.id, st, self.epsilon)

        best = None

        if ac == 0:
            best = self.pickup_nearest(v)
        elif ac == 1:
            best = self.dropoff_nearest(v)
        elif ac == 2:
            best = self.closest_task(v)
        elif ac == 3:
            best = self.oldest_task(v)
        elif ac == 4:
            best = self.best_distance_wait(v)

        if best is None:
            best = self.closest_task(v)

        self.route_agent.remember(v.id, best.id, st, ac)
        return best

    # -----------------------------------------------------
    # TASK SELECTION HELPERS
    # -----------------------------------------------------

    def pickup_nearest(self, v):
        pickups = [o for o in v.orders if o.picked_at is None]
        if not pickups:
            return None
        return min(pickups, key=lambda o: manhattan(v.pos, o.restaurant.pos))

    def dropoff_nearest(self, v):
        drops = [o for o in v.orders if o.picked_at is not None and not o.delivered]
        if not drops:
            return None
        return min(drops, key=lambda o: manhattan(v.pos, o.household.pos))

    def closest_task(self, v):
        if len(v.orders) == 0:
            return None
        return min(v.orders, key=lambda o:
                   manhattan(v.pos, o.restaurant.pos if o.picked_at is None else o.household.pos))

    def oldest_task(self, v):
        return min(v.orders, key=lambda o: o.created_at)

    def best_distance_wait(self, v):
        best = None
        best_score = float("inf")

        for o in v.orders:
            target = o.restaurant.pos if o.picked_at is None else o.household.pos
            d = manhattan(v.pos, target)
            wait = self.time - o.created_at
            score = d + 0.5 * wait
            if score < best_score:
                best_score = score
                best = o

        return best

    # -----------------------------------------------------
    # START NEXT TASK
    # -----------------------------------------------------

    def start_next_task(self, v):

        nxt = self.choose_next_task(v)
        if nxt is None:
            v.state = "idle"
            v.current_task = None
            return

        v.current_task = nxt

        if nxt.picked_at is None:
            v.state = "pickup"
            t = travel_time(v.pos, nxt.restaurant.pos)
            v.distance += manhattan(v.pos, nxt.restaurant.pos)
        else:
            v.state = "dropoff"
            t = travel_time(v.pos, nxt.household.pos)
            v.distance += manhattan(v.pos, nxt.household.pos)

        v.time_remaining = t

    # -----------------------------------------------------
    # MOVE VEHICLES
    # -----------------------------------------------------

    def move_vehicles(self):

        for v in self.vehicles:

            if v.state == "idle":
                v.idle_time += 1
            elif v.state == "pickup":
                v.pickup_time += 1
            elif v.state == "dropoff":
                v.dropoff_time += 1

            if v.time_remaining > 0:
                v.time_remaining -= 1
                continue

            if v.current_task is None:
                continue

            o = v.current_task

            if v.state == "pickup":
                v.pos = o.restaurant.pos
                o.picked_at = self.time

            elif v.state == "dropoff":
                v.pos = o.household.pos
                o.delivered = True
                o.delivered_at = self.time
                self.completed.append(o)

                o.household.busy = False
                o.household.current_order = None

                if self.use_assign_rl and self.assign_agent is not None:
                    dt = o.delivered_at - o.created_at
                    self.assign_agent.update(o, dt, v.distance)

                if self.use_route_rl and self.route_agent is not None:
                    dt = o.delivered_at - o.created_at
                    self.route_agent.update(v.id, o, dt, v.distance)

                v.orders.remove(o)
                v.load -= 1

            v.current_task = None
            self.start_next_task(v)

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------

    def run(self):
        for t in range(SIM_DURATION):
            self.time = t
            self.generate_orders()
            self.assign_orders()
            self.move_vehicles()

        return self.metrics()

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    def metrics(self):
        done = len(self.completed)
        created = len(self.orders)

        if done == 0:
            return {
                "created": created,
                "done": 0,
                "adt": None,
                "late": None,
                "distance": sum(v.distance for v in self.vehicles)
            }

        adt = sum(o.delivered_at - o.created_at for o in self.completed) / done
        late = sum(1 for o in self.completed if (o.delivered_at - o.created_at) > LATE_DEADLINE)
        dist = sum(v.distance for v in self.vehicles)

        return {
            "created": created,
            "done": done,
            "adt": adt,
            "late": late,
            "distance": dist
        }
# ---------------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------------

restaurants = [Restaurant(i, pos) for i, pos in enumerate(restaurants_coords)]
households  = [Household(i, pos)  for i, pos in enumerate(households_coords)]

assign_agent = AssignmentAgent()
route_agent  = RoutingAgent()

def make_vehicles():
    return [
        Vehicle(0, (20, 50)),
        Vehicle(1, (80, 50)),
        Vehicle(2, (20, 50)),
        Vehicle(3, (50, 50)),
        Vehicle(4, (80, 50)),
    ]

def reset_households():
    for h in households:
        h.busy = False
        h.current_order = None

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

history = []
epsilon = EPS_START

for ep in range(RL_EPISODES):

    reset_households()
    vehicles = make_vehicles()

    sim = DVSim(
        restaurants,
        households,
        vehicles,
        assign_agent=assign_agent,
        route_agent=route_agent,
        use_assign_rl=True,
        use_route_rl=True,
        epsilon=epsilon
    )

    result = sim.run()
    history.append(result)

    epsilon = max(EPS_END, epsilon * EPS_DECAY)

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def run_eval(runs=30):
    results = []

    for _ in range(runs):
        reset_households()
        sim = DVSim(
            restaurants,
            households,
            make_vehicles(),
            assign_agent=assign_agent,
            route_agent=route_agent,
            use_assign_rl=True,
            use_route_rl=True,
            epsilon=0.0
        )
        results.append(sim.run())

    # compute averages
    avg = lambda key: sum(r[key] for r in results) / len(results)

    return {
        "created": avg("created"),
        "done": avg("done"),
        "adt": avg("adt"),
        "late": avg("late"),
        "distance": avg("distance")
    }


def run_baseline(runs=30):
    results = []

    for _ in range(runs):
        reset_households()
        sim = DVSim(
            restaurants,
            households,
            make_vehicles(),
            assign_agent=None,
            route_agent=None,
            use_assign_rl=False,
            use_route_rl=False,
            epsilon=0.0
        )
        results.append(sim.run())

    avg = lambda key: sum(r[key] for r in results) / len(results)

    return {
        "created": avg("created"),
        "done": avg("done"),
        "adt": avg("adt"),
        "late": avg("late"),
        "distance": avg("distance")
    }


rl_eval = run_eval()
baseline_eval = run_baseline()

print("RL:", rl_eval)
print("BASELINE:", baseline_eval)

# ---------------------------------------------------------
# GRAPHS
# ---------------------------------------------------------

def smooth(values, w=25):
    if len(values) < w:
        return values
    return np.convolve(values, np.ones(w)/w, mode='valid')

orders_done = [x["done"] for x in history]
adts        = [x["adt"]  if x["adt"] else 0 for x in history]
lates       = [x["late"] if x["late"] else 0 for x in history]
dists       = [x["distance"] for x in history]

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0, 0].plot(smooth(orders_done))
ax[0, 0].set_title("Orders Completed (Smoothed)")

ax[0, 1].plot(smooth(adts))
ax[0, 1].set_title("Average Delivery Time (Smoothed)")

ax[1, 0].plot(smooth(lates))
ax[1, 0].set_title("Late Deliveries (Smoothed)")

ax[1, 1].plot(smooth(dists))
ax[1, 1].set_title("Total Distance (Smoothed)")

plt.tight_layout()
plt.show()
