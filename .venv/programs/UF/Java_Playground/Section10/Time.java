import java.util.Date;
import java.text.SimpleDateFormat;

public class Time {
    private int hour, minute, second;
    SimpleDateFormat  timeFormat = new SimpleDateFormat ("HH:mm:ss");
    SimpleDateFormat  hourFormat = new SimpleDateFormat ("HH");
    SimpleDateFormat  minuteFormat = new SimpleDateFormat ("mm");
    SimpleDateFormat  secondFormat = new SimpleDateFormat("ss");

    Time() {
        Date current = new Date();
        this.hour = Integer.parseInt(hourFormat.format(current));
        this.minute = Integer.parseInt(minuteFormat.format(current));
        this.second = Integer.parseInt(minuteFormat.format(current));
    }

    Time(long ms) {
        Date current = new Date(ms);
        this.hour = Integer.parseInt(hourFormat.format(current));
        this.minute = Integer.parseInt(minuteFormat.format(current));
        this.second = Integer.parseInt(minuteFormat.format(current));
    }

    Time(int hour, int minute, int second) {
        this.hour = hour;
        this.minute = minute;
        this.second = second;
    }

    public void setTime(long ms) {
        Date date = new Date(ms);
        this.hour = Integer.parseInt(hourFormat.format(date));
        this.minute = Integer.parseInt(minuteFormat.format(date));
        this.second = Integer.parseInt(minuteFormat.format(date));
    }

    public String toString() {
        String s = this.hour + ":" + this.minute + ":" + this.second;
        return s;
    }

    public static void main(String[] args) {
        Time t1 = new Time();
        Time t2 = new Time(555550000);
        Time t3 = new Time(5, 23, 55);
        System.out.println(t1);
        System.out.println(t2);
        System.out.println(t3);
    }
}
