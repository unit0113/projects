import java.util.Scanner;

public class StudentScore {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Number of students: ");
        int length = sc.nextInt();
        int[] students = new int[length];

        System.out.print("Input student scores: ");
        int max = 0;
        for (int i = 0; i < length; i++) {
            students[i] = sc.nextInt();
            if (students[i] > max) {
                max = students[i];
            }
        }
        sc.close();
        
        for (int i = 0; i < length; i++) {
            System.out.printf("Student %d score is %d and the grade is %c\n", i, students[i], getGrade(students[i], max));
        }

    }
    
    public static char getGrade(int score, int max) {
        if (score >= max - 10) {
            return 'A';
        }

        else if (score >= max - 20) {
            return 'B';
        }

        else if (score >= max - 30) {
            return 'C';
        }

        else if (score >= max - 40) {
            return 'D';
        }

        return 'F';
    }
}
