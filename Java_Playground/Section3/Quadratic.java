import java.util.Scanner;
import java.lang.Math;

public class Quadratic {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a, b, and c: ");
        float a = sc.nextFloat();
        float b = sc.nextFloat();
        float c = sc.nextFloat();
        sc.close();

        float determinant = b * b - 4 * a * c;
        if (determinant < 0) {
            System.out.println("The equation has no real roots");
        }

        else if (determinant == 0) {
            float root = -b / (2 * a);
            System.out.println("The equation has one root: " + root);
        }

        else {
            double determinantRoot = Math.sqrt(determinant);
            double root1 = (-b + determinantRoot) / (2 * a);
            double root2 = (-b - determinantRoot) / (2 * a);
            System.out.println("The equation has two roots: " + root1 + " and " + root2);
        }
    }
}
