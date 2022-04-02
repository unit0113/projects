import java.util.Scanner;

class Calender {
    int[] daysInMonth;
    int[] monthStartingDay;
    String[] monthsInYear;
    int startingDay;
    int year;

    Calender(int year, int startingDay) {
        this.year = year;
        this.startingDay = startingDay;

        if (year % 4 == 0) {
            this.daysInMonth = new int[]{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        }
        else {
            this.daysInMonth = new int[]{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        }
        this.monthsInYear = new String[]{"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};

        int[] monthStartingDay = new int[12];
        monthStartingDay[0] = this.startingDay;
        for (int i = 1; i < 12; i++) {
            monthStartingDay[i] = (monthStartingDay[i - 1] + daysInMonth[i - 1]) % 7;
        }
        this.monthStartingDay = monthStartingDay;

    }

    public void printMonth(int monthIndex) {
        this.printMonthHeader(monthIndex);
        this.printDays(this.monthStartingDay[monthIndex], this.daysInMonth[monthIndex]);
    }

    public void printMonthHeader(int monthIndex) {
        System.out.println("                  " + this.monthsInYear[monthIndex] + " " + this.year);
        System.out.println("-----------------------------------------------");
        System.out.println("   Sun   Mon  Tues   Wed  Thur   Fri   Sat");     
    }

    public void printDays(int startingDay, int daysInMonth) {
        for (int i = 0; i < startingDay; i++) {
            System.out.print("      ");
        }

        for (int day = 1; day <= daysInMonth; day++) {
            if (day < 10) {
                System.out.printf("     %d", day);
            }
            else {
                System.out.printf("    %d", day);
            }

            if ((startingDay + day) % 7 == 0) {
                System.out.print("\n");
            }
        }

        System.out.println();
        System.out.println();
    }
}

public class DisplayCalander {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a year to display: ");
        int year = sc.nextInt();
        System.out.print("Enter the first weekday of the year (0 for Sunday, 6 for Saturday): ");
        int day = sc.nextInt();        
        sc.close();

        Calender calendar = new Calender(year, day);

        for (int i = 0; i < 12; i++) {
            calendar.printMonth(i);
        }

    }
}
