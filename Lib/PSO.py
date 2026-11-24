# THIS IS THE FILE USED FOR THE PSO ACTUALLY USED IN THE PAPER

import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

GRID_SIZE = 100
SIM_DURATION = 300          # seconds
VEHICLE_SPEED = 1.0         # grid units per second
NUM_VEHICLES = 5
ORDER_RATE = 22             # target number of orders per episode (moderate load)

CAPACITY = 2                # max simultaneous orders per vehicle

# Fitness weights (for scalar fitness)
W_ADT = 1.0                 # average delivery time weight
W_DIST = 0.02               # distance cost weight
W_LATE = 0.2                # proportional lateness penalty weight
LATE_DEADLINE = 60          # seconds

# PSO search bounds for heuristic weights
MIN_W = 0.01
MAX_W = 3.0                 # tighter, easier search

# small noise for scoring to keep behaviour stochastic
SCORE_NOISE = 0.01

# normalisation constants for scoring features
TT_SCALE = 50.0             # typical travel time scale
WAIT_SCALE = 100.0          # typical wait time scale
QUEUE_SCALE = 5.0           # typical queue length scale
LATE_SCALE = 100.0          # typical lateness risk scale

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------
# FIXED RESTAURANTS / HOUSEHOLDS
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

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def travel_time(a, b):
    return max(1, int(manhattan(a, b) / VEHICLE_SPEED))


# ---------------------------------------------------------
# DATA STRUCTURES
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
        self.state = "idle"          # "idle", "pickup", "dropoff"
        self.time_remaining = 0
        self.order = None
        self.tasks = []              # list of ("pickup"/"dropoff", Order)
        self.capacity_left = CAPACITY
        self.distance = 0
        self.idle_time = 0
        self.pickup_time = 0
        self.dropoff_time = 0

    def score_task(self, task, weights, current_time):
        """
        Scoring features:
        - travel time to next stop (pickup or dropoff)
        - current wait time for this order
        - vehicle queue length
        - estimated lateness risk if we serve this order
        All normalised to similar scales, then weighted.
        """
        action, order = task

        if action == "pickup":
            tt_to_next = travel_time(self.pos, order.restaurant.pos)
            pickup_pos = order.restaurant.pos
        else:
            tt_to_next = travel_time(self.pos, order.household.pos)
            pickup_pos = order.restaurant.pos

        wait = current_time - order.created_at

        # simple estimate: go from current -> restaurant -> household
        tt_pickup = travel_time(self.pos, pickup_pos)
        tt_delivery = travel_time(pickup_pos, order.household.pos)
        predicted_finish = current_time + tt_pickup + tt_delivery
        lateness_risk = max(
            0,
            (predicted_finish - (order.created_at + LATE_DEADLINE)) / 60
        )

        queue_len = len(self.tasks)

        # normalise features
        tt_norm = tt_to_next / TT_SCALE
        wait_norm = wait / WAIT_SCALE
        queue_norm = queue_len / QUEUE_SCALE
        late_norm = lateness_risk / LATE_SCALE

        w1, w2, w3, w4 = weights

        score = (
            w1 * tt_norm +
            w2 * wait_norm +
            w3 * queue_norm +
            w4 * late_norm
        )

        # small stochastic noise
        score += random.uniform(0.0, SCORE_NOISE)
        return score


# ---------------------------------------------------------
# ENTITY SETUP + RESET
# ---------------------------------------------------------

restaurants = [Restaurant(i, pos) for i, pos in enumerate(restaurants_coords)]
households = [Household(i, pos) for i, pos in enumerate(households_coords)]


def reset_households():
    for h in households:
        h.busy = False
        h.current_order = None


# ---------------------------------------------------------
# SIMULATION (STOCHASTIC)
# ---------------------------------------------------------

class DVRPSim:
    def __init__(self, restaurants, households, weights):
        self.restaurants = restaurants
        self.households = households
        self.weights = weights

        self.time = 0
        self.next_order_id = 0

        self.vehicles = [
            Vehicle(0, (20, 50)),
            Vehicle(1, (80, 50)),
            Vehicle(2, (20, 50)),
            Vehicle(3, (50, 50)),
            Vehicle(4, (80, 50)),
        ]

        self.orders = []
        self.pending_orders = []
        self.completed = []

        # stochastic order times every run
        self.order_times = sorted(
            random.sample(range(20, SIM_DURATION - 1), ORDER_RATE)
        )

    def generate_order(self):
        if self.time not in self.order_times:
            return

        hh = random.choice(self.households)
        if hh.busy:
            return

        r = random.choice(self.restaurants)
        o = Order(self.next_order_id, r, hh, self.time)
        self.next_order_id += 1

        hh.busy = True
        hh.current_order = o

        self.orders.append(o)
        self.pending_orders.append(o)

    def assign_orders(self):
        for o in list(self.pending_orders):
            available = [v for v in self.vehicles if v.capacity_left > 0]
            if not available:
                continue

            best_v = None
            best_score = float('inf')

            for v in available:
                s = v.score_task(("pickup", o), self.weights, self.time)
                if s < best_score:
                    best_score = s
                    best_v = v

            best_v.capacity_left -= 1
            best_v.tasks.append(("pickup", o))
            best_v.tasks.append(("dropoff", o))
            o.assigned_vehicle = best_v
            self.pending_orders.remove(o)

    def move(self):
        for v in self.vehicles:
            if v.time_remaining > 0:
                v.time_remaining -= 1

                if v.time_remaining > 0:
                    continue

                # finished a movement
                if v.state == "pickup":
                    o = v.order
                    v.pos = o.restaurant.pos
                    o.picked_at = self.time

                    # allow batching if more pickups queued and capacity left
                    has_more_pickups = any(task[0] == "pickup" for task in v.tasks)
                    if has_more_pickups and v.capacity_left > 0:
                        v.state = "idle"
                        v.order = None
                        continue

                    # otherwise go to dropoff
                    v.state = "dropoff"
                    t = travel_time(v.pos, o.household.pos)
                    v.time_remaining = t
                    v.distance += manhattan(v.pos, o.household.pos)
                    continue

                if v.state == "dropoff":
                    o = v.order
                    v.pos = o.household.pos
                    o.delivered = True
                    o.delivered_at = self.time
                    self.completed.append(o)

                    v.capacity_left = min(CAPACITY, v.capacity_left + 1)
                    o.household.busy = False
                    o.household.current_order = None

                    v.order = None
                    v.state = "idle"

            if v.state == "idle" and not v.tasks:
                v.idle_time += 1
                continue

            if v.state == "idle" and v.tasks:
                task, order = v.tasks.pop(0)
                v.order = order

                if task == "pickup":
                    v.state = "pickup"
                    t = travel_time(v.pos, order.restaurant.pos)
                    v.time_remaining = t
                    v.distance += manhattan(v.pos, order.restaurant.pos)
                else:
                    v.state = "dropoff"
                    t = travel_time(v.pos, order.household.pos)
                    v.time_remaining = t
                    v.distance += manhattan(v.pos, order.household.pos)

            if v.state == "pickup":
                v.pickup_time += 1
            elif v.state == "dropoff":
                v.dropoff_time += 1

    def run(self):
        for t in range(SIM_DURATION):
            self.time = t
            self.generate_order()
            self.assign_orders()
            self.move()
        return self.fitness()

    def fitness(self):
        done = len(self.completed)
        if done == 0:
            return 1e9

        adt = sum((o.delivered_at - o.created_at) / 60 for o in self.completed) / done

        lateness_penalty = sum(
            max(0, (o.delivered_at - o.created_at) - LATE_DEADLINE)
            for o in self.completed
        )

        dist = sum(v.distance / 300 for v in self.vehicles)

        return (
            W_ADT * adt +
            W_DIST * dist +
            W_LATE * lateness_penalty
        )


# ---------------------------------------------------------
# PSO IMPLEMENTATION
# ---------------------------------------------------------

class Particle:
    def __init__(self):
        # 4 weights for heuristic: tt_norm, wait_norm, queue_norm, late_norm
        self.position = np.random.uniform(MIN_W, MAX_W, size=4)
        self.velocity = np.zeros(4)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')


class PSO:
    def __init__(self, num_particles, w_start, w_end, c1, c2, iters):
        self.num_particles = num_particles
        self.particles = [Particle() for _ in range(num_particles)]
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_fitness = float('inf')
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.iters = iters

    def current_inertia(self, iter_idx):
        # linear decay from w_start to w_end
        if self.iters <= 1:
            return self.w_end
        alpha = iter_idx / (self.iters - 1)
        return self.w_start + (self.w_end - self.w_start) * alpha

    def evaluate(self, particle):
        fits = []
        for _ in range(3):  # Monte Carlo averaging
            reset_households()
            sim = DVRPSim(restaurants, households, particle.position)
            fits.append(sim.run())

        fit = sum(fits) / len(fits)

        if fit < particle.best_fitness:
            particle.best_fitness = fit
            particle.best_position = particle.position.copy()

        if fit < self.global_best_fitness:
            self.global_best_fitness = fit
            self.global_best_position = particle.position.copy()

        return fit

    def step(self, iter_idx):
        w = self.current_inertia(iter_idx)
        for p in self.particles:
            r1 = np.random.rand(4)
            r2 = np.random.rand(4)

            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)

            p.velocity = w * p.velocity + cognitive + social
            p.position += p.velocity

            p.position = np.clip(p.position, MIN_W, MAX_W)


# ---------------------------------------------------------
# SINGLE-RUN FITNESS HELPER (for baseline inside PSO loop)
# ---------------------------------------------------------

def single_run_fitness(weights):
    reset_households()
    sim = DVRPSim(restaurants, households, weights)
    return sim.run()


# ---------------------------------------------------------
# RUN PSO (with per-iteration baseline)
# ---------------------------------------------------------

NUM_PARTICLES = 30
NUM_ITERS = 80

pso = PSO(num_particles=NUM_PARTICLES,
          w_start=0.9, w_end=0.4,
          c1=1.6, c2=1.6,
          iters=NUM_ITERS)

iter_best_history = []
global_best_history = []
baseline_iter_history = []

baseline_weights = np.array([1.0, 0.0, 0.0, 0.0])  # "distance/time-to-next only" heuristic

for it in range(pso.iters):
    iter_best = float('inf')

    for p in pso.particles:
        f = pso.evaluate(p)
        if f < iter_best:
            iter_best = f

    iter_best_history.append(iter_best)
    global_best_history.append(pso.global_best_fitness)

    baseline_fit = single_run_fitness(baseline_weights)
    baseline_iter_history.append(baseline_fit)

    print(f"iter {it+1} | PSO best this iter: {iter_best:.2f} | "
          f"global best: {pso.global_best_fitness:.2f} | "
          f"baseline: {baseline_fit:.2f}")

    pso.step(it)

best_w = pso.global_best_position
print("\nBEST WEIGHTS:", best_w)


# ---------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------

def evaluate_policy(weights, runs=30):
    fitnesses = []
    created_list = []
    done_list = []
    adt_list = []
    late_list = []
    dist_list = []

    for _ in range(runs):
        reset_households()
        sim = DVRPSim(restaurants, households, weights)
        fit = sim.run()
        fitnesses.append(fit)

        created = len(sim.orders)
        done = len(sim.completed)
        created_list.append(created)
        done_list.append(done)

        if done > 0:
            adt = sum(
                o.delivered_at - o.created_at for o in sim.completed
            ) / done
            late_penalty = sum(
                max(0, (o.delivered_at - o.created_at) - LATE_DEADLINE)
                for o in sim.completed
            )
        else:
            adt = 0.0
            late_penalty = 0.0

        dist = sum(v.distance for v in sim.vehicles)

        adt_list.append(adt)
        late_list.append(late_penalty)
        dist_list.append(dist)

    def avg(x):
        return sum(x) / len(x)

    return {
        "fitness": avg(fitnesses),
        "created": avg(created_list),
        "done": avg(done_list),
        "adt": avg(adt_list),
        "late_penalty": avg(late_list),
        "distance": avg(dist_list),
    }


# ---------------------------------------------------------
# PSO BEST VS BASELINE (multi-run stats)
# ---------------------------------------------------------

print("\nEvaluating PSO best policy over 30 runs...")
pso_metrics = evaluate_policy(best_w, runs=30)

print("\nEvaluating baseline (distance-only heuristic) over 30 runs...")
baseline_metrics = evaluate_policy(baseline_weights, runs=30)

print("\n--- PSO BEST (avg over 30 runs) ---")
print(pso_metrics)

print("\n--- BASELINE (avg over 30 runs) ---")
print(baseline_metrics)


# ---------------------------------------------------------
# PLOTS: ITERATION BEST vs GLOBAL BEST vs BASELINE
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(iter_best_history, label="PSO iteration-best fitness")
plt.plot(global_best_history, label="PSO global-best fitness", linewidth=2)
plt.plot(baseline_iter_history, label="Baseline fitness (per iter)", linestyle="--")
plt.title("PSO vs Baseline Fitness Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness (lower is better)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
