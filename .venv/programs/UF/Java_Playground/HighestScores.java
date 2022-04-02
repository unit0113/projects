import java.util.Scanner;
import java.util.List;
import java.util.ArrayList;


class Person implements Comparable<Person>{
    float score;
    String name;

    public Person(float score, String name) {
        this.score = score;
        this.name = name;
    }

    public float getScore() {
        return this.score;
    }

    public int compareTo(Person p) {
        float score1 = getScore();
        float score2 = p.getScore();

        if (score1 == score2) {
            return 0;
        }
        else if (score1 < score2) {
            return 1;
        }
        else {
            return -1;
        }
    }

    public String toString() {
        String s = this.name + ": " + this.score;
        return s;
    }
}

public class HighestScores {
    public static void main(String[] args) {
        final int TOP = 2;
        Scanner sc = new Scanner(System.in);
        String name;
        float score;
        List<Person> students = new ArrayList<>();
        while (true) {
            System.out.print("Input a name and their score (ex. John 89.5). Enter 'x' to stop: ");
            name = sc.next();
            if (name.equals("x")) {
                break;
            }
            score = sc.nextInt();
            students.add(new Person(score, name));
        }
        sc.close();
        
        sort(students);
        System.out.printf("The highest %d score are:\n", TOP);
        for (int i = 0; i < TOP; i++) {
            System.out.println(students.get(i));
        }

    }

    public static void sort(List<Person> list)
    {
        list.sort((o1, o2) -> o1.compareTo(o2));
    }
}
