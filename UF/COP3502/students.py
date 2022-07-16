class Student:
    def __init__(self, f_name='Louie', l_name='', gpa=1.0) -> None:
        self.f_name = f_name
        self.l_name = l_name
        self.gpa = gpa

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_gpa(self, gpa):
        self.gpa = gpa

    def get_gpa(self):
        return self.gpa


class Course:
    def __init__(self) -> None:
        self.roster = []

    def add_student(self, student):
        self.roster.append(student)

    def course_size(self):
        return len(self.roster)


