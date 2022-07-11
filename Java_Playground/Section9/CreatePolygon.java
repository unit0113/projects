import java.lang.Math;

class RegularPolygon {
    private int n;
    private double side;
    private double x;
    private double y;
    
    RegularPolygon() {
        this.n = 3;
        this.side = 1;
        this.x = 0;
        this.y = 0;
    }

    RegularPolygon(int n) {
        this.n = n;
        this.side = 1;
        this.x = 0;
        this.y = 0;
    }

    RegularPolygon(int n, double side) {
        this.n = n;
        this.side = side;
        this.x = 0;
        this.y = 0;
    }

    RegularPolygon(int n, double side, double x, double y) {
        this.n = n;
        this.side = side;
        this.x = x;
        this.y = y;        
    }

    public void setNumSides(int n) {
        this.n = n;
    }

    public int getNumSides() {
        return this.n;
    }

    public void setSideLength(double side) {
        this.side = side;
    }

    public double getSideLength() {
        return this.side;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getX() {
        return this.x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getY() {
        return this.y;
    }

    public double getPerimeter() {
        return this.n * this.side;
    }

    public double getArea() {
        return (this.n * this.side * this.side) / (4 * Math.tan(Math.PI / this.n));
    }

}

public class CreatePolygon {
    public static void main(String[] args) {
        
    }
}
