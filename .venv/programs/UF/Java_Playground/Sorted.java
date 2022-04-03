import java.util.Scanner;

public class Sorted {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of elements to sort: ");
        int length = sc.nextInt();
        int[] arr = new int[length];
        System.out.print("Enter the array: ");
        for (int i = 0; i < length; i++) {
            arr[i] = sc.nextInt();
        }   
        sc.close();

        if (sorted(arr)) {
            System.out.println("Sorted");
        }
        else {
            System.out.println("Not sorted");
        }
    }

    public static boolean sorted(int[] arr) {
        for (int i = 1; i < arr.length; i++) {
            if (arr[i-1] > arr[i]) {
                return false;
            }
        }
        return true;
    }
}