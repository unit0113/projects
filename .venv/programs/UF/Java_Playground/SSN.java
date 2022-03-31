import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SSN {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter an SSN: ");
        String ssn = sc.next();   
        sc.close();

        Pattern pattern = Pattern.compile("\\d{3}-\\d{2}-\\d{4}");
        Matcher matcher = pattern.matcher(ssn);

        if (matcher.find()) {
            System.out.printf("%s is a valid SSN", ssn);
        }

        else {
            System.out.printf("%s is not a valid SSN", ssn);
        }

    }
}
