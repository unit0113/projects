import java.util.Scanner;

public class NumberUnique {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] numbers = new int[10];
        System.out.print("Enter 10 numbers: ");
        int input;
        int numFilled = 0;
        for (int i = 0; i < 10; i++) {
            input = sc.nextInt();
            if (doesNotContain(numbers, input)) {
                numbers[numFilled] = input;
                numFilled++;
            }
        }
        sc.close();
        System.out.printf("Unique numbers: %d\n", numFilled);
        System.out.print("The unique numbers are: ");

        for (int i = 0; i < numFilled; i++) {
            System.out.printf("%d ", numbers[i]);
        }

    }

    public static boolean doesNotContain(int[] arr, int key) {
        for (int num : arr) {
            if (num == key) {
                return false;
            }
        }
        return true;
    }
}