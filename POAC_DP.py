import xml.etree.ElementTree as ET
import random
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

##############################
# ACO-DP 통합 Full Framework
##############################

NUM_ANTS = 30
GENERATIONS = 100
EVAPORATION_RATE = 0.1
PHEROMONE_INIT = 1.0
PHEROMONE_BOOST = 50.0

def aco_dp_framework(data):
    courses = data['courses']
    chromosome_template = data['chromosome']
    total_slots = data['total_slots']

    course_list = [(entry['department_id'], entry['course_id'], entry['teacher_id']) for entry in chromosome_template]
    pheromone = [PHEROMONE_INIT for _ in range(total_slots)]

    best_schedule = None
    best_fitness = float('inf')
    fitness_progress = []

    for generation in range(GENERATIONS):
        ant_solutions = []
        for _ in range(NUM_ANTS):
            schedule = construct_solution(course_list, data, pheromone)
            assignment = dp_room_assignment(schedule, data)
            if check_hard_constraints_logic(schedule, assignment, data):
                fit = fitness(schedule, data)
                ant_solutions.append((schedule, fit))
                if fit < best_fitness:
                    best_fitness = fit
                    best_schedule = schedule

        # pheromone evaporation
        pheromone = [p * (1 - EVAPORATION_RATE) for p in pheromone]

        # pheromone update
        for schedule, fit in ant_solutions:
            for slots in schedule.values():
                for slot in slots:
                    pheromone[slot] += PHEROMONE_BOOST / (1 + fit)

        print(f"Gen {generation} - Best Score: {best_fitness}")
        fitness_progress.append(best_fitness)

    plt.plot(fitness_progress)
    plt.title("Best Score per Generation (ACO)")
    plt.xlabel("Generation")
    plt.ylabel("Best Score")
    plt.grid(True)
    plt.show()

    final_room_assignment = dp_room_assignment(best_schedule, data)
    check_hard_constraints(best_schedule, final_room_assignment, data)
    export_schedule(best_schedule, final_room_assignment, data)
    return best_schedule, final_room_assignment


def construct_solution(course_list, data, pheromone):
    schedule = {}
    total_slots = data['total_slots']

    for dept_id, course_id, teacher_id in course_list:
        lectures = data['courses'][course_id].get('lectures', 1)
        slots = []
        available = list(range(total_slots))
        for _ in range(lectures):
            probs = [pheromone[s] for s in available]
            total = sum(probs)
            if total == 0:
                chosen = random.choice(available)
            else:
                probs = [p / total for p in probs]
                chosen = random.choices(available, weights=probs, k=1)[0]
            slots.append(chosen)
            available.remove(chosen)
        schedule[(dept_id, course_id)] = slots
    return schedule

def export_schedule(schedule, room_assignment, data, filename='schedule.csv'):
    records = []
    for (dept_id, course_id), slots in schedule.items():
        rooms = room_assignment[(dept_id, course_id)]
        for i in range(len(slots)):
            records.append({
                'Department': dept_id,
                'Course': course_id,
                'Slot': slots[i],
                'Room': rooms[i],
                'Students': data['courses'][course_id].get('students', 0)
            })
    df = pd.DataFrame(records)
    df.to_csv(filename, index=False)
    print(f"Schedule exported to {filename}")


def generate_random_schedule(course_list, data):
    schedule = {}
    for dept_id, course_id, teacher_id in course_list:
        total_slots = data['total_slots']
        lectures = data['courses'][course_id].get('lectures', 1)
        slots = random.sample(range(total_slots), min(lectures, total_slots))
        schedule[(dept_id, course_id)] = slots
    return schedule


def fitness(schedule, data):
    penalty = 0
    for (dept_id, course_id), slots in schedule.items():
        days = [((slot // data['slotsPerDay']) % data['nrDays']) for slot in slots]
        day_counts = defaultdict(int)
        for d in days:
            day_counts[d] += 1
        for count in day_counts.values():
            if count > 1:
                penalty += (count - 1) * 5

    daily_load = defaultdict(int)
    for (dept_id, course_id), slots in schedule.items():
        for slot in slots:
            day = ((slot // data['slotsPerDay']) % data['nrDays'])
            daily_load[(dept_id, day)] += 1
    for count in daily_load.values():
        if count > 4:
            penalty += (count - 4) * 10

    teacher_load = defaultdict(int)
    for (dept_id, course_id), slots in schedule.items():
        teacher = data['courses'][course_id].get('teacher', 'NO_TEACHER')
        for slot in slots:
            day = ((slot // data['slotsPerDay']) % data['nrDays'])
            if teacher != 'NO_TEACHER':
                teacher_load[(teacher, day)] += 1
    for count in teacher_load.values():
        if count > 3:
            penalty += (count - 3) * 8

    return penalty


def dp_room_assignment(schedule, data):
    rooms = list(data['rooms'].items())
    slot_to_rooms = defaultdict(list)

    for room_id, info in rooms:
        for slot in info['available_slots']:
            slot_to_rooms[slot].append((room_id, info['capacity']))

    room_assignment = {}
    for (dept_id, course_id), slots in schedule.items():
        students = data['courses'][course_id].get('students', 0)
        assigned_rooms = []
        for slot in slots:
            possible = [(room_id, cap) for (room_id, cap) in slot_to_rooms.get(slot, []) if cap >= students]
            if possible:
                best_room = min(possible, key=lambda x: x[1]-students)[0]
            else:
                best_room = None
            assigned_rooms.append(best_room)
        room_assignment[(dept_id, course_id)] = assigned_rooms

    return room_assignment

def check_hard_constraints(schedule, room_assignment, data):
    print("\n--- Hard Constraint Violation Report ---")

    capacity_violations = 0
    for (dept_id, course_id), rooms in room_assignment.items():
        students = data['courses'][course_id].get('students', 0)
        for room_id in rooms:
            if room_id is None:
                print(f"Room not assigned for course {course_id}")
                capacity_violations += 1
            elif data['rooms'][room_id]['capacity'] < students:
                print(f"Capacity violation: course {course_id}, room {room_id}")
                capacity_violations += 1

    overlap_counter = defaultdict(list)
    for (dept_id, course_id), slots in schedule.items():
        for i, slot in enumerate(slots):
            room = room_assignment[(dept_id, course_id)][i]
            if room is not None:
                overlap_counter[(slot, room)].append((dept_id, course_id))

    overlap_violations = 0
    for key, assigned_courses in overlap_counter.items():
        if len(assigned_courses) > 1:
            print(f"Room conflict at slot {key[0]}, room {key[1]}: courses {assigned_courses}")
            overlap_violations += 1

    if capacity_violations == 0 and overlap_violations == 0:
        print("No hard constraint violations found!")

    print("--- End of Report ---\n")

def check_hard_constraints_logic(schedule, room_assignment, data):
    for (dept_id, course_id), rooms in room_assignment.items():
        students = data['courses'][course_id].get('students', 0)
        for room_id in rooms:
            # 교실 수용인원 부족
            if room_id is None or data['rooms'][room_id]['capacity'] < students:
                return False

    overlap_counter = defaultdict(int)
    for (dept_id, course_id), slots in schedule.items():
        for i, slot in enumerate(slots):
            room = room_assignment[(dept_id, course_id)][i]
            if room is not None:
                overlap_counter[(slot, room)] += 1
                # 교실 중복 사용
                if overlap_counter[(slot, room)] > 1:
                    return False

    return True
