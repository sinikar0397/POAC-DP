import xml.etree.ElementTree as ET
import random
import copy
from collections import defaultdict

##############################
# UniTime 통합 Safe Parser
##############################

def parse_unitime(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    # 1️⃣ Parse rooms
    rooms = {}
    for room_elem in root.find('rooms').findall('room'):
        room_id = room_elem.attrib['id']
        capacity = int(room_elem.attrib.get('capacity', 0))
        available_slots = set()

        unavailable = [False] * 100000
        for unav in room_elem.findall('unavailable'):
            days = unav.attrib['days']
            start = int(unav.attrib['start'])
            length = int(unav.attrib['length'])
            weeks = unav.attrib['weeks']
            for day_idx, day_bit in enumerate(days):
                if day_bit == '1':
                    for week_idx, week_bit in enumerate(weeks):
                        if week_bit == '1':
                            for l in range(length):
                                slot = (week_idx * 7 + day_idx) * 288 + start + l
                                unavailable[slot] = True
        for i in range(len(unavailable)):
            if not unavailable[i]:
                available_slots.add(i)

        rooms[room_id] = {
            'capacity': capacity,
            'available_slots': list(available_slots)
        }

    # 2️⃣ Parse courses & classes
    courses = {}
    for course_elem in root.find('courses').findall('course'):
        course_id = course_elem.attrib['id']
        class_elems = course_elem.findall('.//class')
        total_lectures = len(class_elems)
        student_sum = sum(int(c.attrib.get('limit', 0)) for c in class_elems)

        courses[course_id] = {
            'teacher': 'NO_TEACHER',
            'lectures': total_lectures if total_lectures > 0 else 1,
            'min_days': 1,
            'students': student_sum
        }

    # 3️⃣ Parse curricula as students
    curricula = defaultdict(list)
    for student_elem in root.find('students').findall('student'):
        student_id = student_elem.attrib['id']
        for course_ref in student_elem.findall('course'):
            course_id = course_ref.attrib['id']
            curricula[student_id].append(course_id)

    # 4️⃣ Build chromosome
    chromosome = []
    for dept_id, course_list in curricula.items():
        for course_id in course_list:
            entry = {
                'department_id': dept_id,
                'course_id': course_id,
                'teacher_id': 'NO_TEACHER',
                'merged_class_id': None
            }
            chromosome.append(entry)

    # 5️⃣ Extract time structure
    nrDays = int(root.attrib.get('nrDays', 7))
    slotsPerDay = int(root.attrib.get('slotsPerDay', 288))
    nrWeeks = int(root.attrib.get('nrWeeks', 15))
    total_slots = nrDays * slotsPerDay * nrWeeks

    return {
        'nrDays': nrDays,
        'slotsPerDay': slotsPerDay,
        'nrWeeks': nrWeeks,
        'total_slots': total_slots,
        'rooms': rooms,
        'courses': courses,
        'curricula': curricula,
        'chromosome': chromosome
    }