import java.util.Scanner;

public class IntPalindrome {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Please enter an integer: ");
        int intialInt = sc.nextInt();
        sc.close();

        int reverseInt = reverse(intialInt);

        if (isPalindrome(intialInt, reverseInt)) {
            System.out.printf("The number %d is a palindrome", intialInt);
        }

        else {
            System.out.printf("%d is not a palindrome", intialInt);
        }
        
    }

    public static int reverse(int intialInt) {
        String result = "";

        while (intialInt > 0) {
            result += intialInt % 10;
            intialInt /= 10;
        }

        return Integer.parseInt(result);
    }

    public static boolean isPalindrome(int num1, int num2) {
        return num1 == num2;
        
    }
}
