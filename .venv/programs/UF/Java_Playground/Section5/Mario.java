import java.util.Scanner;

public class Mario {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("How high would you like your pyramid?");
        int input = sc.nextInt();
        sc.close();

        for (int i = 1; i <= input; i++) {
            String line = "";
            for (int j = input - i; j >= 0; j--) {
                line += " ";
            }

            while (line.length() <= input) {
                line += "#";
            }

            System.out.println(line);
        }
    }
}