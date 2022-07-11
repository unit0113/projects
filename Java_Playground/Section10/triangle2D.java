import java.lang.Math;

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

class Triangle {
    private Point[] points;
    private double area;

    Triangle(Point p1, Point p2, Point p3) {
        points = new Point[] {p1, p2, p3};
        area = calcArea();
    }

    public Point[] getPoints() {
        return this.points;
    }

    private double calcArea() {
        double part1 = points[0].getX() * (points[1].getY() - points[2].getY());
        double part2 = points[1].getX() * (points[2].getY() - points[0].getY());
        double part3 = points[2].getX() * (points[0].getY() - points[1].getY());
        area = 0.5 * Math.abs(part1 + part2 + part3);
        return area;
    }

    public boolean contains(Point p) {
        double area1 = (new Triangle(points[0], points[1], p)).calcArea();
        double area2 = (new Triangle(points[1], points[2], p)).calcArea();
        double area3 = (new Triangle(points[2], points[0], p)).calcArea();
        double sumArea = area1 + area2 + area3;
        if (Math.abs(this.area - sumArea) <= 0.0000001) {
            return true;
        }
        return false;
    }

    public boolean contains(Triangle t) {
        Point[] newPoints = t.getPoints();
        for (Point p : newPoints) {
            if (!this.contains(p)) {
                return false;
            }
        }
        return true;
    }
}

public class triangle2D {
    public static void main(String[] args) {
        Point p1 = new Point(0, 0);
        Point p2 = new Point(4, 4);
        Point p3 = new Point(8, 0);
        Triangle t1 = new Triangle(p1, p2, p3);

        Point pIn = new Point(4, 2);
        Point pOut = new Point(8, 4);
        Point pIn2 = new Point(6, 1);
        Triangle tOut = new Triangle(pIn, pOut, pIn2);

        Point pIn3 = new Point(3, 1);
        Triangle tIn = new Triangle(pIn, pIn2, pIn3);

        System.out.println(t1.contains(pIn));
        System.out.println(t1.contains(pIn2));
        System.out.println(t1.contains(pIn3));
        System.out.println(t1.contains(pOut));

        System.out.println(t1.contains(tOut));
        System.out.println(t1.contains(tIn));
    }
}
