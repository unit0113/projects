import java.util.Scanner;

public class Pyramid {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int height = 0;
        while (height < 1 || height > 15){
            System.out.print("Input an integer (1-15): ");
            height = sc.nextInt();
        }
        sc.close();

        for (int i = 1; i <= height; i++) {
            int spaces = 2 * (height - i);
            String leadingSpaces = "";
            for (int l = 0; l < spaces; l++) {
                leadingSpaces += " ";
            }
            System.out.print(leadingSpaces);            
            
            for (int j = i; j > 1; j--) {
                System.out.printf("%d ", j);
            }

            System.out.print("1 ");

            for (int k = 2; k <= i; k++) {
                System.out.printf("%d ", k);
            }

            System.out.print("\n");
        }

    }
}