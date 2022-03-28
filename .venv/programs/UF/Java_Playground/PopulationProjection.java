import java.util.Scanner;
import java.lang.Math;

public class PopulationProjection {
    public static void main(String[] args) {
        final float BIRTHS = 7.0f;
        final float DEATHS = 13.0f;
        final float MIGRANT = 45.0f;
        final int POPULATION = 312_032_486;

        float birthsPerSecond = 1 / BIRTHS;
        float deathsPerSecond = 1 / DEATHS;
        float migrantsPerSecond = 1 / MIGRANT;

        Scanner sc = new Scanner(System.in);
        System.out.println("Calculat population how many years out?");
        int years = sc.nextInt();
        sc.close();

        float seconds = years * 365 * 24 * 60 * 60.0f;
        int finalPop = POPULATION + Math.round(seconds * (birthsPerSecond - deathsPerSecond + migrantsPerSecond));

        System.out.println("The final population is " + finalPop);

    }
}
