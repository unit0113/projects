package plc.homework;

import java.util.regex.Pattern;

/**
 * Contains {@link Pattern} constants, which are compiled regular expressions.
 * See the assignment page for resources on regexes as needed.
 */
public class Regex {

    public static final Pattern
            EMAIL = Pattern.compile("[A-Za-z0-9._-]+@[A-Za-z0-9-]*\\.[a-z]{2,3}"),
            ODD_STRINGS = Pattern.compile(".{11}|.{13}|.{15}|.{17}|.{19}"),
            INTEGER_LIST = Pattern.compile("^\\[\\s*([^,]\\s*(\\d+){0,1}(,\\s*\\d+)*\\s*)?\\]$"),
            DECIMAL = Pattern.compile("^-?(0|[1-9][0-9]*)\\.\\d*$"),
            STRING = Pattern.compile("^\\\"([^\\\\]|\\\\[bnrt'\\\"\\\\])*\\\"$");
}
