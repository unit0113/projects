import java.lang.Math;


class Point {
    double x;
    double y;

    Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public String toString() {
        String s = "(" + this.x + ", " + this.y + ")";
        return s;
    }
}


public class PointsOnCircle {
    public static void main(String[] args) {
        final double RADIUS = 40;
        Point point1 = randPoint(RADIUS);
        Point point2 = randPoint(RADIUS);
        Point point3 = randPoint(RADIUS);

        double a = calcDist(point1, point2);
        double b = calcDist(point1, point3);
        double c = calcDist(point3, point2);

        double angle1 = calcAngle(a, b, c);
        double angle2 = calcAngle(b, c, a);
        double angle3 = calcAngle(c, a, b);

        System.out.printf("The angles between points " + point1 + ", " + point2 + ", " + point3 + " are %.3f, %.3f, and %.3f", angle1, angle2, angle3);
    }
    
    
    public static Point randPoint(double radius) {
        double angle = Math.random() * (2 * Math.PI);
        double x = radius * Math.cos(angle);
        double y = radius * Math.sin(angle);
        return new Point(x, y);
    }


    public static double calcDist(Point point1, Point point2) {
        return Math.sqrt(Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2));
        
    }


    public static double calcAngle(double a, double b, double c) {
        double angle = (a*a - b*b - c*c) / (-2 * b * c);
        return Math.toDegrees(Math.acos(angle));
        
    }
}
