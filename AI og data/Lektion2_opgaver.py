import sqlite3

def create_database():
    conn = sqlite3.connect('school.db')
    print("Opened database successfully")

    cur = conn.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS Students
                   (student_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    major TEXT NOT NULL)''')
    print("Table 'Students' created successfully")

    cur.execute('''CREATE TABLE IF NOT EXISTS Courses
                   (course_id INTEGER PRIMARY KEY,
                    course_name TEXT NOT NULL,
                    instructor TEXT NOT NULL)''')
    print("Table 'Courses' created successfully")

    cur.execute('''CREATE TABLE IF NOT EXISTS Enrollments
                   (enrollment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL,
                    course_id INTEGER NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES Students(student_id),
                    FOREIGN KEY (course_id) REFERENCES Courses(course_id))''')
    print("Table 'Enrollments' created successfully")
    
    students_data = [
        (1, 'Flemming', 'AI'),
        (2, 'Trine', 'Biologi'),
        (3, 'Mogens', 'Kemi'),
        (4, 'Jonas', 'Fysik'),
        (5, 'Jørgen', 'Engelsk'),
        (6, 'Lars', 'AI'),
        (7, 'Mette', 'Biologi'),
        (8, 'Søren', 'Engelsk')
    ]

    cur.executemany('INSERT INTO Students (student_id, name, major) VALUES (?, ?, ?)', students_data)
    print("Inserted records into 'Students' successfully")

    courses_data = [
        (11, 'AI og data', 'Alan Turing'),
        (12, 'Molekulær biologi', 'Rosalind Franklin'),
        (13, 'Organisk kemi', 'Marie Curie'),
        (14, 'tyngdeacceleration', 'Isaac Newton'),
        (15, 'Klassiske digte', 'William Shakespeare')
    ]
    cur.executemany('INSERT INTO Courses (course_id, course_name, instructor) VALUES (?, ?, ?)', courses_data)
    print("Inserted records into 'Courses' successfully")

    enrollment_data = [
        (1, 11),
        (1, 12),
        (2, 12),
        (3, 13),
        (4, 14),
        (5, 15),
        (5, 14),
        (6, 11),
        (7, 12),
        (8, 15)
    ]
    cur.executemany('INSERT INTO Enrollments (student_id, course_id) VALUES (?, ?)', enrollment_data)
    print("Inserted records into 'Enrollments' successfully")

    conn.commit()
    conn.close()
    print("Closed database successfully")
#create_database()

def fetch_courses_by_student(student_id):
    with sqlite3.connect('school.db') as conn:
        cur = conn.cursor()
        query = '''
            SELECT Courses.course_id, Courses.course_name, Courses.instructor
            FROM Courses
            JOIN Enrollments ON Courses.course_id = Enrollments.course_id
            WHERE Enrollments.student_id = ?;
        '''
        cur.execute(query, (student_id,))
        courses = cur.fetchall()
    return courses

def fetch_students_by_course(course_id):
    with sqlite3.connect('school.db') as conn:
        cur = conn.cursor()
        query = '''
            SELECT Students.student_id, Students.name, Students.major
            FROM Students
            JOIN Enrollments ON Students.student_id = Enrollments.student_id
            WHERE Enrollments.course_id = ?;
        '''
        cur.execute(query, (course_id,))
        students = cur.fetchall()
    return students

student_id = 4  
course_id = 14  

print("Courses for student ID:", student_id)
for course in fetch_courses_by_student(student_id):
    print(course)

print("\nStudents in course ID:", course_id)
for student in fetch_students_by_course(course_id):
    print(student)
