import xml.etree.ElementTree as ET
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

##############################
# POAC-DP (ACO + DP) Framework (Optimized with CDF Sampling + Noise)
##############################

# ACO hyperparameters
NUM_ANTS = 30
GENERATIONS = 100
EVAPORATION_RATE = 0.1
PHEROMONE_INIT = 2.0
PHEROMONE_BOOST = 15
MUTATION_BASE = 0.15
MUTATION_BOOST = 0.15

# Quick feasibility check before expensive ops
def quick_hard_constraint_check(schedule, data):
    slot_counts = defaultdict(int)
    total_rooms = len(data['rooms'])
    for key, slots in schedule.items():
        for s in slots:
            slot_counts[s] += 1
            if slot_counts[s] > total_rooms:
                return False
    return True

# Construct ant solution using optimized CDF-based pheromone sampling + noise

def construct_solution(course_list, data, pheromone):
    schedule = {}
    total_slots = data['total_slots']

    noise = np.random.uniform(0.95, 1.05, size=pheromone.shape)
    noisy_pheromone = pheromone * noise

    cdf = np.cumsum(noisy_pheromone)
    max_ph = cdf[-1]

    for dept, course, teacher in course_list:
        lectures = data['courses'][course].get('lectures', 1)
        chosen = set()
        while len(chosen) < lectures:
            if max_ph <= 0:
                idx = random.randrange(total_slots)
            else:
                u = random.random() * max_ph
                idx = int(np.searchsorted(cdf, u, side='right'))
            chosen.add(idx)
        schedule[(dept, course)] = list(chosen)
    return schedule

# Mutation: random replacement with adaptive rate
def mutate_schedule(schedule, data, gen):
    total_slots = data['total_slots']
    adaptive_rate = MUTATION_BASE + MUTATION_BOOST * (gen / GENERATIONS)
    for key, slots in schedule.items():
        for i in range(len(slots)):
            if random.random() < adaptive_rate:
                schedule[key][i] = random.randint(0, total_slots - 1)

# Greedy DP room assignment minimizing wasted seats
def dp_room_assignment(schedule, data):
    slot_to_rooms = defaultdict(list)
    for rid, info in data['rooms'].items():
        for s in info['available_slots']:
            slot_to_rooms[s].append((rid, info['capacity']))
    assignment = {}
    for key, slots in schedule.items():
        need = data['courses'][key[1]].get('students', 0)
        sel = []
        for s in slots:
            candidates = [(r, cap) for r, cap in slot_to_rooms.get(s, []) if cap >= need]
            sel.append(min(candidates, key=lambda x: x[1] - need)[0] if candidates else None)
        assignment[key] = sel
    return assignment

# Fitness calculation (soft constraints)
def fitness(schedule, data):
    penalty = 0
    for key, slots in schedule.items():
        days = [(s // data['slotsPerDay']) % data['nrDays'] for s in slots]
        cnt = defaultdict(int)
        for d in days:
            cnt[d] += 1
        for c in cnt.values():
            if c > 1:
                penalty += (c - 1) * 5
    load = defaultdict(int)
    for key, slots in schedule.items():
        for s in slots:
            d = (s // data['slotsPerDay']) % data['nrDays']
            load[(key[0], d)] += 1
    for c in load.values():
        if c > 4:
            penalty += (c - 4) * 10
    tload = defaultdict(int)
    for key, slots in schedule.items():
        t = data['courses'][key[1]].get('teacher', 'NO')
        if t != 'NO':
            for s in slots:
                d = (s // data['slotsPerDay']) % data['nrDays']
                tload[(t, d)] += 1
    for c in tload.values():
        if c > 3:
            penalty += (c - 3) * 8
    return penalty

# Hard constraint check/report
def check_hard_constraints(schedule, assignment, data):
    cap_v = 0
    overlap_v = 0
    for key, rooms in assignment.items():
        need = data['courses'][key[1]].get('students', 0)
        for r in rooms:
            if r is None or data['rooms'][r]['capacity'] < need:
                cap_v += 1
    sr = defaultdict(list)
    for key, slots in schedule.items():
        for i, s in enumerate(slots):
            r = assignment[key][i]
            if r is not None:
                sr[(s, r)].append(key)
    for v in sr.values():
        if len(v) > 1:
            overlap_v += 1
    if cap_v == 0 and overlap_v == 0:
        print('No hard constraint violations.')
    else:
        print(f'Capacity violations: {cap_v}, Overlap violations: {overlap_v}')

# Exports
def export_schedule(schedule, assignment, data, fname):
    os.makedirs('./result/schedule', exist_ok=True)
    rec = []
    for key, slots in schedule.items():
        for s, r in zip(slots, assignment[key]):
            rec.append({
                'Dept': key[0], 'Course': key[1], 'Slot': s,
                'Room': r, 'Students': data['courses'][key[1]].get('students', 0)
            })
    pd.DataFrame(rec).to_csv(f'./result/schedule/POAC-DP_{fname}.csv', index=False)
    print(f'Schedule exported to ./result/schedule/POAC-DP_{fname}.csv')

def export_fitness(log, fname):
    os.makedirs('./result/fitness', exist_ok=True)
    pd.DataFrame({'Gen': list(range(len(log))), 'BestFit': log}).to_csv(f'./result/fitness/POAC-DP_{fname}.csv', index=False)
    print(f'Fitness exported to ./result/fitness/POAC-DP_{fname}.csv')

# Main ACO-DP framework
def poac_dp_framework(data, fname):
    chrom = data['chromosome']
    course_list = [(e['department_id'], e['course_id'], e['teacher_id']) for e in chrom]

    total_slots = data['total_slots']
    pheromone = np.full(total_slots, PHEROMONE_INIT, dtype=float)
    best_sched = None
    best_fit = np.inf
    log = []

    for gen in range(GENERATIONS):
        sols = []
        for _ in range(NUM_ANTS):
            sched = construct_solution(course_list, data, pheromone)
            mutate_schedule(sched, data, gen)
            if not quick_hard_constraint_check(sched, data):
                continue
            fit = fitness(sched, data)
            sols.append((sched, fit))

        if sols:
            sols.sort(key=lambda x: x[1])
            elite_sols = sols[:max(1, int(len(sols) * 0.2))]  # Top 20%
            sched0, fit0 = elite_sols[0]
            if fit0 < best_fit:
                best_fit = fit0
                best_sched = sched0

            pheromone *= (1 - EVAPORATION_RATE)
            for sched, fit in elite_sols:
                boost = (PHEROMONE_BOOST * 1000) / (1 + fit)
                for slots in sched.values():
                    for s in slots:
                        pheromone[s] += boost

        log.append(best_fit)
        print(f'Gen {gen} POAC best={best_fit}')

    plt.plot(log)
    plt.title('POAC-DP Best Fit')
    plt.xlabel('Gen'); plt.ylabel('Fit'); plt.grid()
    os.makedirs('./result/image', exist_ok=True)
    plt.savefig(f'./result/image/POAC-DP_{fname}.png')

    export_fitness(log, fname)
    final_assign = dp_room_assignment(best_sched, data)
    check_hard_constraints(best_sched, final_assign, data)
    export_schedule(best_sched, final_assign, data, fname)
    return best_sched, final_assign
