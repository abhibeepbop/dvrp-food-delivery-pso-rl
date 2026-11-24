import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

GRID_SIZE = 100
SIM_DURATION = 300

VEHICLE_SPEED = 1.0
VEHICLE_CAPACITY = 2
NUM_VEHICLES = 5

BASE_ORDER_RATE = 22

RL_EPISODES = 30000

# tuned hyperparameters
EPS_START = 0.40
EPS_END = 0.02
EPS_DECAY = 0.998

ALPHA = 0.12  # tuned learning rate
GAMMA = 0.0  # because you're using final reward only

DISTANCE_COST = 0.07
LATE_PENALTY = 40
LATE_DEADLINE = 60

random.seed(7)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def travel_time(a, b):
    return max(1, int(manhattan(a, b) / VEHICLE_SPEED))


# ---------------------------------------------------------
# ENTITIES
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
        self.state = "idle"
        self.time_remaining = 0
        self.orders = []
        self.current_task = None
        self.load = 0
        self.distance = 0


# ---------------------------------------------------------
# UNIFIED RL AGENT
# ---------------------------------------------------------

class UnifiedRLAgent:

    def __init__(self):

        # state buckets
        self.dist_buckets = 4  # 0–3
        self.load_buckets = 3  # 0–2+
        self.wait_buckets = 4
        self.pending_buckets = 3

        self.num_states = (
                self.dist_buckets *
                self.load_buckets *
                self.wait_buckets *
                self.pending_buckets
        )

        # routing actions (5)
        # 0 pickup-nearest
        # 1 dropoff-nearest
        # 2 overall-nearest
        # 3 FIFO
        # 4 best distance/wait score
        self.num_actions = 5

        # one Q-table per vehicle
        self.Q = np.zeros((NUM_VEHICLES, self.num_states, self.num_actions))

        self.memory = {}  # order_id → (veh_id, state, action)

    # bucket helpers

    def bucket_dist(self, d):
        if d <= 10: return 0
        if d <= 25: return 1
        if d <= 50: return 2
        return 3

    def bucket_load(self, load):
        return min(load, 2)

    def bucket_wait(self, w):
        if w <= 20: return 0
        if w <= 40: return 1
        if w <= 80: return 2
        return 3

    def bucket_pending(self, p):
        if p <= 2: return 0
        if p <= 5: return 1
        return 2

    # state builder

    def build_state(self, vehicle, pending_count, current_time):

        if vehicle.orders:
            # use nearest task distance
            d0 = min(manhattan(vehicle.pos,
                               o.restaurant.pos if o.picked_at is None else o.household.pos)
                     for o in vehicle.orders)
        else:
            d0 = 999

        dist_b = self.bucket_dist(d0)
        load_b = self.bucket_load(vehicle.load)

        if vehicle.orders:
            w = max(current_time - o.created_at for o in vehicle.orders)
        else:
            w = 0

        wait_b = self.bucket_wait(w)
        pend_b = self.bucket_pending(pending_count)

        idx = (
                dist_b +
                self.dist_buckets * (
                        load_b +
                        self.load_buckets * (
                                wait_b +
                                self.wait_buckets * pend_b
                        )
                )
        )

        return idx

    # epsilon-greedy

    def select_action(self, veh_id, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        return np.argmax(self.Q[veh_id, state])

    # store for update

    def remember(self, order_id, veh_id, state, action):
        self.memory[order_id] = (veh_id, state, action)

    # update after delivery

    def update(self, order, dt, veh_dist):
        if order.id not in self.memory:
            return

        veh_id, st, ac = self.memory.pop(order.id)

        late_flag = 1 if dt > LATE_DEADLINE else 0

        reward = -(
                (dt / 60) +
                (veh_dist / 300) +
                (3 * late_flag)
        )

        old = self.Q[veh_id, st, ac]
        self.Q[veh_id, st, ac] = old + ALPHA * (reward - old)


# ---------------------------------------------------------
# SIMULATION (unchanged except routing uses unified RL)
# ---------------------------------------------------------

class DVSim:

    def __init__(self, restaurants, households, vehicles,
                 rl_agent=None, use_rl=False, epsilon=0.1):

        self.restaurants = restaurants
        self.households = households
        self.vehicles = vehicles

        self.rl_agent = rl_agent
        self.use_rl = use_rl
        self.epsilon = epsilon

        self.time = 0
        self.next_order_id = 0

        self.orders = []
        self.pending = []
        self.completed = []

        self.order_times = sorted(random.sample(range(10, SIM_DURATION),
                                                BASE_ORDER_RATE))

    def generate_orders(self):
        if self.time not in self.order_times:
            return

        hh = random.choice(self.households)
        if hh.busy:
            free = [h for h in self.households if not h.busy]
            if not free:
                return
            hh = random.choice(free)

        r = random.choice(self.restaurants)
        o = Order(self.next_order_id, r, hh, self.time)
        self.next_order_id += 1

        hh.busy = True
        hh.current_order = o

        self.orders.append(o)
        self.pending.append(o)

    def assign_orders(self):

        for o in list(self.pending):

            available = [v for v in self.vehicles if v.load < VEHICLE_CAPACITY]
            if not available:
                continue

            # baseline: nearest vehicle
            chosen = min(available,
                         key=lambda vv: manhattan(vv.pos, o.restaurant.pos))

            chosen.orders.append(o)
            chosen.load += 1
            o.assigned_vehicle = chosen
            self.pending.remove(o)

            if chosen.state == "idle" and chosen.time_remaining == 0:
                self.start_next_task(chosen)

    def choose_next_action(self, v):

        if not v.orders:
            return None, None

        pending_count = len([o for o in self.orders if not o.delivered])
        st = self.rl_agent.build_state(v, pending_count, self.time)

        ac = self.rl_agent.select_action(v.id, st, self.epsilon)

        # decode action
        if ac == 0:
            task = min((o for o in v.orders if o.picked_at is None),
                       key=lambda o: manhattan(v.pos, o.restaurant.pos),
                       default=None)
        elif ac == 1:
            task = min((o for o in v.orders if o.picked_at is not None and not o.delivered),
                       key=lambda o: manhattan(v.pos, o.household.pos),
                       default=None)
        elif ac == 2:
            task = min(v.orders,
                       key=lambda o: manhattan(v.pos,
                                               o.restaurant.pos if o.picked_at is None
                                               else o.household.pos))
        elif ac == 3:
            task = min(v.orders, key=lambda o: o.created_at)
        elif ac == 4:
            best, best_score = None, float("inf")
            for o in v.orders:
                d = manhattan(v.pos,
                              o.restaurant.pos if o.picked_at is None
                              else o.household.pos)
                w = self.time - o.created_at
                s = d + 0.5 * w
                if s < best_score:
                    best_score = s
                    best = o
            task = best

        if task is None:
            # fallback
            task = min(v.orders,
                       key=lambda o: manhattan(v.pos,
                                               o.restaurant.pos if o.picked_at is None
                                               else o.household.pos))

        self.rl_agent.remember(task.id, v.id, st, ac)
        return task, ac

    def start_next_task(self, v):

        if not self.use_rl:
            # fallback baseline: nearest
            if not v.orders:
                return
            nxt = min(v.orders,
                      key=lambda o: manhattan(v.pos,
                                              o.restaurant.pos if o.picked_at is None
                                              else o.household.pos))
        else:
            nxt, _ = self.choose_next_action(v)

        if nxt is None:
            v.state = "idle"
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

    def move_vehicles(self):

        for v in self.vehicles:

            if v.time_remaining > 0:
                v.time_remaining -= 1
                continue

            if v.current_task is None:
                continue

            o = v.current_task

            # pickup done
            if v.state == "pickup":
                v.pos = o.restaurant.pos
                o.picked_at = self.time

            # dropoff done
            elif v.state == "dropoff":
                v.pos = o.household.pos
                o.delivered = True
                o.delivered_at = self.time
                self.completed.append(o)

                o.household.busy = False
                o.household.current_order = None

                # SAFE RL UPDATE
                if self.use_rl and self.rl_agent is not None:
                    dt = o.delivered_at - o.created_at
                    self.rl_agent.update(o, dt, v.distance)

                v.orders.remove(o)
                v.load -= 1

            v.current_task = None
            self.start_next_task(v)

    def run(self):
        for t in range(SIM_DURATION):
            self.time = t
            self.generate_orders()
            self.assign_orders()
            self.move_vehicles()

        return self.metrics()

    def metrics(self):
        done = len(self.completed)
        created = len(self.orders)

        if done == 0:
            return {"created": created, "done": 0, "adt": None,
                    "late": None, "distance": sum(v.distance for v in self.vehicles)}

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
# TRAINING
# ---------------------------------------------------------

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

restaurants = [Restaurant(i, pos) for i, pos in enumerate(restaurants_coords)]
households = [Household(i, pos) for i, pos in enumerate(households_coords)]


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


rl_agent = UnifiedRLAgent()

history = []
epsilon = EPS_START

for ep in range(RL_EPISODES):
    reset_households()
    vehicles = make_vehicles()

    sim = DVSim(
        restaurants,
        households,
        vehicles,
        rl_agent=rl_agent,
        use_rl=True,
        epsilon=epsilon
    )

    res = sim.run()
    history.append(res)

    epsilon = max(EPS_END, epsilon * EPS_DECAY)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def run_eval(runs=30):
    results = []
    for _ in range(runs):
        reset_households()
        sim = DVSim(restaurants, households, make_vehicles(),
                    rl_agent=rl_agent, use_rl=True, epsilon=0.0)
        results.append(sim.run())

    avg = lambda k: sum(r[k] for r in results) / len(results)

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
        sim = DVSim(restaurants, households, make_vehicles(),
                    rl_agent=None, use_rl=False, epsilon=0)
        results.append(sim.run())

    avg = lambda k: sum(r[k] for r in results) / len(results)

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
# PLOTS
# ---------------------------------------------------------

def smooth(x, w=30):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


orders_done = [x["done"] for x in history]
adts = [x["adt"] if x["adt"] else 0 for x in history]
lates = [x["late"] if x["late"] else 0 for x in history]
dists = [x["distance"] for x in history]

fig, ax = plt.subplots(2, 2, figsize=(14, 10))

ax[0][0].plot(smooth(orders_done));
ax[0][0].set_title("Orders Completed")
ax[0][1].plot(smooth(adts));
ax[0][1].set_title("ADT")
ax[1][0].plot(smooth(lates));
ax[1][0].set_title("Late Deliveries")
ax[1][1].plot(smooth(dists));
ax[1][1].set_title("Distance")

plt.tight_layout()
plt.show()
