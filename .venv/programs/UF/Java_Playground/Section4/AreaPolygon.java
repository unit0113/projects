import java.util.Scanner;
import java.lang.Math;


public class AreaPolygon {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        int sides = 0;
        while (sides < 3) {
            System.out.print("Enter the number of sides (greater than two): ");
        sides = sc.nextInt();
        }

        System.out.print("Enter the length of each side: ");
        double length = sc.nextDouble();
        sc.close();

        double area = sides * length * length / (4 * Math.tan(Math.PI / sides));

        System.out.printf("The area of the polygon is %.3f", area);
    }
}
