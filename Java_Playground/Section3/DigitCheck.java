import java.util.Scanner;


public class DigitCheck {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String number = "";
        while (number.length() != 9) {
            System.out.print("Enter the first nine digits of an ISBN as an integer: ");
            number = sc.next();
        }
        sc.close();

        int digit;
        int sum = 0;
        int modNumber = Integer.parseInt(number);
        for (int i = 9; i > 0; i--) {
            digit = modNumber % 10;
            sum += digit * i;
            modNumber /= 10;
        }

        int checkDigit = sum % 11;
        if (checkDigit == 10){
            System.out.println("The ISBN-10 number of " + number + " is " + number + "X");
        }
        else {
            System.out.println("The ISBN-10 number of " + number + " is " + number + checkDigit);
        }
    }
}
