import java.util.Scanner;

class Unicode {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a character: ");
        char c = sc.next().charAt(0);
        sc.close();

        int ascii = (int) c;
        System.out.printf("The ASCII value of %c is %d", c, ascii);
    }
}