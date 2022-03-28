import java.util.Scanner;

public class RandMonth {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int monthInt = 0;
        while (monthInt < 1 | monthInt > 12){
            System.out.print("Input an integer for the month (1-12): ");
            monthInt = sc.nextInt();
        }
        sc.close();

        String[] months = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "Novemeber", "December"};
        System.out.println(months[monthInt - 1]);

    }
}
