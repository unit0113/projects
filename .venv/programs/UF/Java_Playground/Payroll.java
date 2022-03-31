import java.util.Scanner;

public class Payroll {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter employee's name: ");
        String name = sc.next();
        System.out.print("Enter number of hours worked this week: ");
        int hours = sc.nextInt();
        System.out.print("Enter hourly pay rate: ");
        float pay = sc.nextFloat();
        float gross = pay * hours;
        System.out.print("Enter federal tax withholding rate: ");
        float federal = sc.nextFloat();
        System.out.print("Enter state tax withholding rate: ");
        float state = sc.nextFloat();
        float deduction = gross * (federal + state);
        float net = gross - deduction;
        sc.close();

        System.out.printf("Employee Name: %s\n", name);
        System.out.printf("Hours Worked: %.1f\n", (float) hours);
        System.out.printf("Pay Rate: $%.2f\n", pay);
        System.out.printf("Gross Pay: $%.2f\n", gross);
        System.out.println("Deductions:");
        System.out.printf("    Federal Withholding (%.1f%%): $%.2f\n", federal * 100, federal * gross);
        System.out.printf("    State Withholding (%.1f%%): $%.2f\n", state * 100, state * gross);
        System.out.printf("    Total Deductions: $%.2f\n", deduction);
        System.out.printf("Net Pay: $%.2f", net);
    }
}
