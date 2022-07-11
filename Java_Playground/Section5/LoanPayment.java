import java.util.Scanner;
import java.lang.Math;

public class LoanPayment {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int loanAmount = 0;
        while (loanAmount < 1){
            System.out.print("Enter the total loan amount: ");
            loanAmount = sc.nextInt();
        }

        double loanDuration = 0;
        while (loanDuration < 1){
            System.out.print("Enter the length of the loan in years: ");
            loanDuration = 12 * sc.nextInt();
        }
        sc.close();

        printHeader();

        double monthlyPayment;
        for (double interestRate = 5.0; interestRate <= 8; interestRate += 0.125) {
            monthlyPayment = calcPayment(loanAmount, loanDuration, interestRate);
            System.out.printf("%.3f%%              %.2f              %.2f\n", interestRate, monthlyPayment, monthlyPayment * loanDuration);            
        }

    }

    private static void printHeader() {
        System.out.print("Interest Rate      ");
        System.out.print("Monthly Payment    ");
        System.out.print("Total Payment\n");
    }

    private static double calcPayment(int loanAmount, double loanDuration, double interestRate) {
        double rateMod = (interestRate / 12) / 100;
        double numerator = Math.pow(1 + rateMod, loanDuration);
        double payment = loanAmount * rateMod * numerator / (numerator - 1);
        return payment;
    }
}
