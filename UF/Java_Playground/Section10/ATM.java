import java.util.Scanner;

class Account {
    private double balance;
    private Scanner sc = new Scanner(System.in);
    
    Account() {
        this.balance = 100;
    }

    Account(double intialDeposit) {
        this.balance = intialDeposit;
    }

    public void checkBalance() {
        System.out.printf("Current Balance: $%.2f\n", this.balance);
    }

    public void deposit() {
        System.out.print("Enter amount to deposit: ");
        this.balance += sc.nextDouble();
    }

    public void withdraw() {
        System.out.print("Enter amount to withdraw: ");
        double amount = sc.nextDouble();
        if (balance > amount) {
            this.balance -= amount;
        }
        else {
            System.out.println("Insufficient funds");
        }
        
    }

    public void mainMenu() {
        int option = 1;
        while (option > 0 && option < 4) {
            this.printMenu();
            option = sc.nextInt();
            switch(option) {
                case 1:
                    this.checkBalance();
                    break;
                case 2:
                    this.withdraw();
                    break;
                case 3:
                    this.deposit();
                    break;
            }
        }
    }

    public void printMenu(){
        System.out.println("Main Menu");
        System.out.println("1: Check Balance");
        System.out.println("2: Withdraw Funds");
        System.out.println("3: Deposit Funds");
        System.out.println("4: Exit");
    }


}

public class ATM {
    public static void main(String[] args) {
        Account[] accounts = new Account[10];
        for (int i = 0; i < accounts.length; i++) {
            Account account = new Account();
            accounts[i] = account;
        }

        Scanner sc = new Scanner(System.in);
        int id;
        while (true) {
            System.out.print("Enter an ID ");
            id = sc.nextInt();
            if (id >= 0 && id < 10) {
                accounts[id].mainMenu();
            }
            else {
                break;
            }
            

        }
        sc.close();
        
    }
}
