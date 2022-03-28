import java.util.ArrayList;
import java.util.Scanner;


public class LinearSort {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ArrayList<Integer> array = new ArrayList<Integer>();
        int nextNum = 0;
        while (true) {
            System.out.print("Enter an integer to add to the array, input -1 to stop adding: ");
            nextNum = sc.nextInt();
            if (nextNum == -1) {
                break;
            }
            array.add(nextNum);
        }
        sc.close();
        System.out.println("The array to be sorted is: " + array);

        int minIndex;
        int tmp;
        for (int i = 0; i < array.size(); i++) {
            minIndex = i;
            for (int number : array.subList(i+1, array.size())) {
                if (number < array.get(minIndex)) {
                    minIndex = array.indexOf(number);
                }
            }
            tmp = array.get(minIndex);
            array.set(minIndex, array.get(i));
            array.set(i, tmp);
        }
        System.out.println("The sorted array is: " + array);
    }
}
