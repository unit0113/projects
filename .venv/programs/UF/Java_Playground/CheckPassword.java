import java.util.Scanner;

public class CheckPassword {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a password: ");
        String password = sc.next();
        sc.close();

        if (has8Char(password) && onlyLettersNumbers(password) && has2digits(password)) {
            System.out.println("Valid password");
        }
        else {
            System.out.println("Invalid password");
        }
    }

    public static boolean has8Char(String password) {
        return password.length() >= 8;
    }

    public static boolean onlyLettersNumbers(String password) {
        for (char c : password.toCharArray()) {
            if (!(Character.isDigit(c) || Character.isLetter(c))) {
                return false;
                }
            }
        return true;
    }

    public static boolean has2digits(String password) {
        int countInt = 0;
        for (char c : password.toCharArray()) {
            if (Character.isDigit(c)) {
                countInt++;
                if (countInt == 2) {
                    return true;
                }
            }
        }
        return false;
    }
}
