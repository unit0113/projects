import java.util.Scanner;

public class SumSeries {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Input a number to sum towards: ");
        int input = sc.nextInt();
        sc.close();

        int sum = 0;
        for (int i = 1; i <= input; i++){
            sum += i;
        }

        System.out.println("The sum of the numbers up to " + input + " is " + sum);

    }
}
