public class Lockers {
    public static void main(String[] args) {
        boolean[] lockers = new boolean[101];
        for (int student = 1; student <= 100; student++) {
            for (int locker = student; locker <= 100; locker += student){
                lockers[locker] = !lockers[locker];
            }
        }

        for (int locker = 1; locker <= 100; locker++) {
            if (lockers[locker]) {
                System.out.print(locker + " ");
            }
            
        }

    }
}
