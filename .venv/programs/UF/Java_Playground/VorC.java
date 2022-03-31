import java.util.Scanner;

public class VorC {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a character: ");
        char c = sc.next().charAt(0);
        sc.close();

        final String VOWELS = "aeiou";
        final String CONSONANTS = "bcdfghjklmnpqrstvwxyz";

        char cLower = Character.toLowerCase(c);
        
        if (VOWELS.contains(String.valueOf(cLower))) {
            System.out.printf("%c is a vowel", c);
        }

        else if (CONSONANTS.contains(String.valueOf(cLower))) {
            System.out.printf("%c is a constonant", c);
        }

        else {
            System.out.printf("%c is an invalid input", c);
        }
    }
}
