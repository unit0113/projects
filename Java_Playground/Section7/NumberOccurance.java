import java.util.Scanner;

public class NumberOccurance {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] numbers = new int[101];
        System.out.print("Insert integers from 1-100: ");
        int input;
        while (sc.hasNext()) {
            input = sc.nextInt();
            if (input == 0) {
                break;
            }
            numbers[input]++;
        }
        sc.close();

        for (int num = 1; num <= 100; num++) {
            if (numbers[num] > 0) {
                System.out.printf("%d occurs %d times\n", num, numbers[num]);
            }
        }
    }
}
