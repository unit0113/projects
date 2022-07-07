import java.util.Scanner;

public class Cipher {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter a message to encypt: ");
        String message = sc.next();
        System.out.print("Enter a cypher code: ");
        int cypher = sc.nextInt() % 26;    
        sc.close(); 

        String result = "";
        char[] messageChar = message.toCharArray();
        for (char c : messageChar) {
            int ascii = cypher + (int) c;
            result += (char) ascii;
        }

        System.out.println("The encrypted message is: " + result);
    }
}
