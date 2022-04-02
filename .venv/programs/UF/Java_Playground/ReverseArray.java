import java.util.Arrays;

public class ReverseArray {
    public static void main(String[] args) {
        int[] arr1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int[] arr2 = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        reverse(arr1);
        reverse(arr2);
        System.out.println(Arrays.toString(arr1));
        System.out.println(Arrays.toString(arr2));
    }

    public static void reverse(int[] array) {
        int tmp;
        int length = array.length;
        for (int i = 0; i < length / 2; i++) {
            tmp = array[length - i - 1];
            array[length - i - 1] = array[i];
            array[i] = tmp;
        }
    }
}
