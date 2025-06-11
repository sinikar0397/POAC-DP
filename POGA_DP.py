import xml.etree.ElementTree as ET
import random
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd # 데이터 저장을 위해서만 사용

##############################
# POGA-DP 통합 Full Framework
##############################

POPULATION_SIZE = 30
GENERATIONS = 100
MUTATION_RATE = 0.1
SWAP_RATE = 0.3
MAX_MUTATION_ATTEMPTS = 10


def poga_dp_framework(data, fname):
    courses = data['courses']
    chromosome_template = data['chromosome']
    rooms = data['rooms']

    course_list = [(entry['department_id'], entry['course_id'], entry['teacher_id']) for entry in chromosome_template]
    population = [generate_random_schedule(course_list, data) for _ in range(POPULATION_SIZE)]

    best_scores = []

    for generation in range(GENERATIONS):
        fitness_scores = [(schedule, fitness(schedule, data)) for schedule in population]
        fitness_scores.sort(key=lambda x: x[1])
        best_schedule, best_score = fitness_scores[0]
        print(f"Gen {generation} - Best Score: {best_score}")
        best_scores.append(best_score)

        new_population = [best_schedule]
        while len(new_population) < POPULATION_SIZE:
            parent = tournament_selection(fitness_scores)
            child = mutate(parent, data)
            new_population.append(child)
        population = new_population

    plt.plot(best_scores)
    plt.title("Best Score per Generation")
    plt.xlabel("Generation")
    plt.ylabel("Best Score")
    plt.grid(True)
    plt.show()
    plt.savefig('./result/image/'+fname+'.png')

    final_schedule = population[0]
    final_room_assignment = dp_room_assignment(final_schedule, data)
    check_hard_constraints(final_schedule, final_room_assignment, data)
    export_schedule(final_schedule, final_room_assignment, data)
    return final_schedule, final_room_assignment


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


def tournament_selection(fitness_scores, k=3):
    selected = random.sample(fitness_scores, k)
    selected.sort(key=lambda x: x[1])
    return copy.deepcopy(selected[0][0])


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


def mutate(schedule, data):
    original_fitness = fitness(schedule, data)
    for _ in range(MAX_MUTATION_ATTEMPTS):
        new_schedule = copy.deepcopy(schedule)

        # 기존 random mutation
        for key in schedule:
            if random.random() < MUTATION_RATE:
                total_slots = data['total_slots']
                lectures = data['courses'][key[1]].get('lectures', 1)
                new_slots = random.sample(range(total_slots), min(lectures, total_slots))
                new_schedule[key] = new_slots

        # 추가 swap mutation
        if random.random() < SWAP_RATE:
            depts = list(set([key[0] for key in schedule.keys()]))
            dept = random.choice(depts)
            dept_courses = [key for key in schedule if key[0] == dept]
            if len(dept_courses) >= 2:
                a, b = random.sample(dept_courses, 2)
                idx_a = random.randint(0, len(new_schedule[a]) - 1)
                idx_b = random.randint(0, len(new_schedule[b]) - 1)
                new_schedule[a][idx_a], new_schedule[b][idx_b] = new_schedule[b][idx_b], new_schedule[a][idx_a]

        if fitness(new_schedule, data) <= original_fitness and quick_hard_constraint_check(new_schedule, data):
            return new_schedule
    return schedule


def quick_hard_constraint_check(schedule, data):
    slot_counter = defaultdict(int)
    for (dept_id, course_id), slots in schedule.items():
        for slot in slots:
            slot_counter[slot] += 1
            if slot_counter[slot] > len(data['rooms']):
                return False
    return True
