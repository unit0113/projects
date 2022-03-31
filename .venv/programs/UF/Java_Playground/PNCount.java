import java.util.Scanner;

public class PNCount {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int sum, input, pos, neg;
        sum = input = pos = neg = 0;
        while(true) {
            System.out.print("Enter an integer, stop entering by inputing 0: ");
            input = sc.nextInt();

            if (input == 0) {
                break;
            }
            else if (input > 0) {
                pos++;
            }
            else {
                neg++;
            }
            sum += input;
        }
        sc.close();

        float average = sum / (pos + neg);

        System.out.printf("Number of positives: %d\n", pos);
        System.out.printf("Number of negatives: %d\n", neg);
        System.out.printf("Total: %d\n", sum);
        System.out.printf("Average: %.2f\n", average);
    }
}
