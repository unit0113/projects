import java.util.Arrays;
import java.util.Scanner;

public class FutureDate {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("What day is today? ");
        String today = sc.next();
        System.out.print("How many days out? ");
        int future = sc.nextInt();
        sc.close();

        String[] days = {"Sunday", "Monday", "Tuesday", "Wednsday", "Thursday", "Friday", "Saturday"};
        int dayInt = Arrays.asList(days).indexOf(today);
        int newDay = (dayInt + future) % 7;
        System.out.println("Today is " + today + " and the future day is " + days[newDay]);
    }
}