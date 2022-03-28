import java.util.Scanner;

public class Tips {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Input bill: ");
        float bill = sc.nextFloat();
        System.out.print("Input gratuity percentage: ");
        float gratuity = sc.nextFloat() / 100;
        sc.close();

        float tips = gratuity * bill;
        float totalBill = bill + tips;
        System.out.printf("The gratuity is $%.2f and the total is $%.2f", tips, totalBill);
        
    }
}
