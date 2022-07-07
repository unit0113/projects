import java.util.Scanner;

public class HeadsTails {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int state = -1;
        while (state < 0 || state > 511) {
            System.out.print("Enter a number between 0 and 511: ");
            state = sc.nextInt();
        }
        sc.close();

        String binaryState = Integer.toBinaryString(state);
        binaryState = String.format("%9s", binaryState).replace(' ', '0');
        
        char[][] board = new char[3][3];
        int i = 0;
        int j = 0;
        for (char c : binaryState.toCharArray()) {
            board[i][j] = c == '0' ? 'H' : 'T';
            j++;
            if (j == 3) {
                j = 0;
                i++;
            }
        }
        
        for (i = 0; i < board.length; i++) {
            for (j = 0; j < board[0].length; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }
}
