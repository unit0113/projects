import java.util.Scanner;

public class YearsCalculator {
    public static void main(String[] args) {
        final int MINUTES_PER_DAY = 60 * 24;
        final int MINUTES_PER_YEAR = MINUTES_PER_DAY * 365;
        Scanner sc = new Scanner(System.in);
        System.out.print("Input the number of minutes: ");
        long minutes = sc.nextLong();
        sc.close();

        int years = 0, days = 0;
        long modMinutes = minutes;
        while (modMinutes >= MINUTES_PER_YEAR) {
            years++;
            modMinutes -= MINUTES_PER_YEAR;
        }

        while (modMinutes >= MINUTES_PER_DAY) {
            days++;
            modMinutes -= MINUTES_PER_DAY;
        }

        System.out.println(minutes + " minutes is approximately " + years + " and " + days + " days");
    }
}
