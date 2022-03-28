import java.util.Scanner;

public class Interest {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("How much are you saving per month? ");
        int savingsPerMonth = sc.nextInt();
        System.out.print("What is your annual interest rate in percent? ");
        float interestRate = 1 + (sc.nextFloat() / (100 * 12));
        System.out.print("Calculate how many months out? ");
        int months = sc.nextInt();
        sc.close();

        float sum = 0.0f;
        for (int i = 0; i < months; i++) {
            sum += savingsPerMonth;
            sum *= interestRate;
        }

        System.out.printf("After " + months + " months the account value is $%.2f", sum);
        
    }
}