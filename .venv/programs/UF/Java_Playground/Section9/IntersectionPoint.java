import java.util.Scanner;

class Point {
    private double x;
    private double y;

    Point(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }
}

class Line {
    private double a, b, c;

    Line(Point p1, Point p2) {
        double slope = (p2.getY() - p1.getY()) / (p2.getX() - p1.getX());
        this.c = -slope * p1.getX() + p1.getY();
        this.a = -slope;
        this.b = 1;
    }
    
    public double getA() {
        return this.a;
    }

    public double getB() {
        return this.b;
    }

    public double getC() {
        return this.c;
    }

    
}

class LinearEquation {
    private double a, b, c, d, e, f, x, y;

    LinearEquation(Line l1, Line l2) {
        this.a = l1.getA();
        this.b = l1.getB();
        this.e = l1.getC();
        this.c = l2.getA();
        this.d = l2.getB();
        this.f = l2.getC();
        this.x = (this.e * this.d - this.b * this.f) / (this.a * this.d - this.b * this.c);
        this.y = (this.a * this.f - this.e * this.c) / (this.a * this.d - this.b * this.c);
    }

    public double getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }

    public boolean isSolvable() {
        return (this.a * this.d - this.b * this.c) != 0;
    }

    public void printIntersection() {
        String s = "(" + this.getX() + ", " + this.getY() + ")";
        System.out.println(s);
    }
}

public class IntersectionPoint {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter four points in the format X Y: ");

        double x, y;
        x = sc.nextDouble();
        y = sc.nextDouble();
        Point p1 = new Point(x, y);
        x = sc.nextDouble();
        y = sc.nextDouble();
        Point p2 = new Point(x, y);
        Line l1 = new Line(p1, p2);

        x = sc.nextDouble();
        y = sc.nextDouble();
        Point p3 = new Point(x, y);
        x = sc.nextDouble();
        y = sc.nextDouble();
        Point p4 = new Point(x, y);
        Line l2 = new Line(p3, p4);

        sc.close();

        LinearEquation le = new LinearEquation(l1, l2);
        if (le.isSolvable()) {
            le.printIntersection();
        }


    }
}