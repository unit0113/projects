import java.util.Date;
import java.lang.Math;

public class DisplayDate {
    public static void main(String[] args) {
        for (double i = 4; i <= 13; i++) {
            System.out.println(new Date((long) Math.pow(10, i)));
            // new Date((double) Math.pow(10, i))
        }
    }
}
