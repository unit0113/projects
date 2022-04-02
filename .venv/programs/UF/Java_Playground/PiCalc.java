import java.lang.Math;

public class PiCalc {
    public static void main(String[] args) {
        int[] powers = {10_000, 20_000, 100_000};
        
        double sum;
        for (int power : powers) {
            sum = 0;
            for (int i = 1; i <= power; i++) {
                sum += Math.pow(-1, i + 1) / (2 * i - 1);
            }
            System.out.printf("The value of pi calculated up to %d digits is %.6f\n", power, 4 * sum);
            
        }
    }
}
