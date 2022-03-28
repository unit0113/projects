import java.util.Scanner;

public class DigitSum {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Input an integer: ");
        int input = sc.nextInt();
        sc.close();

        int sum = 0;
        int modInput = input;
        while (modInput != 0) {
            sum += modInput % 10;
            modInput /= 10;
        }

        System.out.println("The sum of the digits of " + input + " is " + sum);

    }
}
