import java.util.Scanner;
import java.lang.Math;

public class GreatCircle {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the coordinates of the first point: ");
        double x1 = Math.toRadians(sc.nextDouble());
        double y1 = Math.toRadians(sc.nextDouble());
        System.out.print("Enter the coordinates of the second point: ");
        double x2 = Math.toRadians(sc.nextDouble());
        double y2 = Math.toRadians(sc.nextDouble());
        sc.close();

        float radiusEarth = 6_371.01f;
        double distance = radiusEarth * Math.acos(Math.sin(x1) * Math.sin(x2) + Math.cos(x1) * Math.cos(x2) * Math.cos(y1 - y2));
        System.out.printf("The distance between the points is %.3f km", distance);
    }
}