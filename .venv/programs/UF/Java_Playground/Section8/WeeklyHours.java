import java.util.Scanner;

public class WeeklyHours {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of employees: ");
        int height = sc.nextInt();
        int width = 8;
        int[][] matrix = new int[height][width];
        int sum, input;
        System.out.println("Enter the hours worked per day: ");
        for (int i = 0; i < height; i++) {
            sum = 0;
            for (int j = 0; j < width - 1; j++) {
                input = sc.nextInt();
                matrix[i][j] = input;
                sum += input;
            }
            matrix[i][width-1] = sum;
        }
        sc.close();

        sort(matrix);
        printMatrix(matrix);
        
    }

    public static void sort(int[][] matrix) {
        int maxIndex;
        int[] tmp;
        for (int i = 0; i < matrix.length; i++) {
            maxIndex = i;
            for (int checkIndex = i + 1; checkIndex < matrix.length; checkIndex++) {
                if (matrix[checkIndex][7] > matrix[maxIndex][7]) {
                    maxIndex = checkIndex;
                }
            }
            tmp = matrix[maxIndex];
            matrix[maxIndex] = matrix[i];
            matrix[i] = tmp;
        }
    }

    public static void printMatrix(int[][] matrix) {
        System.out.println("Employee    Sun  Mon  Tue  Wed  Thu  Fri  Sat  Tot");
        for (int i = 0; i < matrix.length; i++) {
            System.out.printf("Employee %d   ", i);
            for (int j = 0; j < matrix[0].length; j++){
                System.out.printf("%d    ", matrix[i][j]);
            }
            System.out.println();
        }
    }
}
