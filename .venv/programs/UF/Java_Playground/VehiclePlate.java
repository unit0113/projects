import java.lang.Math;
import java.util.Random;

public class VehiclePlate {
    public static void main(String[] args) {
        String plate = "";
        while (plate.length() < 3) {
            plate += getLetter();
        }

        Random random = new Random();
        while (plate.length() < 7) {
            plate += getNumber(random);
        }

        System.out.printf("Generated plate: %s", plate);
    }

    public static char getLetter() {
        int ascii = (int) (Math.random() * (90 - 65) + 65);
        return (char) ascii;
    }

    public static int getNumber(Random random) {
        return random.nextInt(10);
    }
}
