public class Tuition {
    public static void main(String[] args) {
        int tuition = 10_000;
        float yearlyIncrease = 1.05f;
        int yearsOut = 10;

        for (int i = 0; i < yearsOut; i++) {
            tuition *= yearlyIncrease;
        }

        System.out.printf("The yearly tuition in %d years will be $%d\n", yearsOut, tuition);
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += tuition;
            tuition *= yearlyIncrease;
        }
        System.out.printf("The total four year cost will be $%d", sum);
    }
}