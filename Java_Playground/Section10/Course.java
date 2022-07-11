class Person {
    private String name;
    private String id;

    Person(String name, String id) {
        this.name = name;
        this.id = id;
    }

    public String getName() {
        return this.name;
    }

    public String getID() {
        return this.id;
    }
}

public class Course {
    private int maxStudents;
    private int maxInstructors;
    private int numStudents;
    private int numInstructors;
    private Person[] students;
    private Person[] instructors;

    Course() {
        this(100, 3);
    }

    Course(int maxStudents, int maxInstructors) {
        this.maxStudents = maxStudents;
        this.maxInstructors = maxInstructors;
        this.students = new Person[maxStudents];
        this.instructors = new Person[maxInstructors];  
        this.numStudents = 0;    
        this.numInstructors = 0;  
    }

    public int getStudents() {
        return this.numStudents;
    }

    public void addStudent(String name, String id) {
        Person student = new Person(name, id);
        students[numStudents] = student;
        numStudents++;
        if (numStudents == maxStudents) {
            students = this.expandArray(students);
        }
    }

    public void addStudent(Person student) {
        students[numStudents] = student;
        numStudents++;
        if (numStudents == maxStudents) {
            students = this.expandArray(students);
        }
    }

    public int findStudent(String id) {
        for (int i = 0; i < numStudents; i++) {
            if (students[i].getID().equals(id)) {
                return i;
            }
        }
        return -1;
    }

    public void dropStudent(String id) {
        int studentIndex = this.findStudent(id);
        if (studentIndex == -1) {
            System.out.println("Student not found");
            return;
        }

        for (int i = studentIndex; i < numStudents; i++) {
            students[i] = students[i+1];
            if (students[i+1] == null) {
                break;
            }
        }

        if (numStudents == maxStudents){
            students[maxStudents-1] = null;
        }
        numStudents--;
    }

    public void clear() {
		students = new Person[maxStudents];
		numStudents = 0;
	}

    public Person[] expandArray(Person[] arr) {
        Person[] newArr = new Person[arr.length * 2];
        System.arraycopy(arr, 0, newArr, 0, arr.length);
        return newArr;
    }

    public int getInstructors() {
        return this.numInstructors;
    }

    public void addInstructor(String name, String id) {
        if (numInstructors == maxInstructors) {
            System.out.println("Warning: Maximum number of instructors assigned to class");
            return;
        }
        Person instructor = new Person(name, id);
        instructors[numStudents] = instructor;
        numInstructors++;
    }

    public static void main(String[] args) {
        Course course1 = new Course();
        Person stud1 = new Person("John", "1");
        Person stud2 = new Person("Mark", "2");
        Person stud3 = new Person("Bob", "3");
        Person stud4 = new Person("April", "4");
        Person stud5 = new Person("Megan", "5");
        Person stud6 = new Person("Lexi", "6");
        course1.addStudent(stud1);
        course1.addStudent(stud2);
        course1.addStudent(stud3);
        course1.addStudent(stud4);
        course1.addStudent(stud5);
        course1.addStudent(stud6);
        course1.addStudent("Billy", "7");
        System.out.println(course1.getStudents());
        course1.dropStudent("8");
        System.out.println(course1.students[6]);
        course1.dropStudent("7");
        System.out.println(course1.getStudents());
        System.out.println(course1.students[6]);
        Course course2 = new Course(2, 1);
        course2.addStudent(stud1);
        System.out.println(course2.students.length);
        course2.addStudent(stud2);
        System.out.println(course2.students.length);
    }
}