import java.util.Scanner;

public class SumByColm {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter a 3x4 matrix row by row");
        int height = 3;
        int width = 4;
        double[][] matrix = new double[height][width];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix[i][j] = sc.nextDouble();
            }
        }
        sc.close();

        double sum;
        for (int colm = 0; colm < width; colm++) {
            sum = 0;
            for (int row = 0; row < height; row++) {
                sum += matrix[row][colm];
            }
            System.out.printf("Sum of column %d is %.2f\n", colm, sum);
        }
        
    }
}
