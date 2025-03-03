package plc.homework;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Contains JUnit tests for {@link Regex}. A framework of the test structure 
 * is provided, you will fill in the remaining pieces.
 *
 * To run tests, either click the run icon on the left margin, which can be used
 * to run all tests or only a specific test. You should make sure your tests are
 * run through IntelliJ (File > Settings > Build, Execution, Deployment > Build
 * Tools > Gradle > Run tests using <em>IntelliJ IDEA</em>). This ensures the
 * name and inputs for the tests are displayed correctly in the run window.
 */
public class RegexTests {

    /**
     * This is a parameterized test for the {@link Regex#EMAIL} regex. The
     * {@link ParameterizedTest} annotation defines this method as a
     * parameterized test, and {@link MethodSource} tells JUnit to look for the
     * static method {@link #testEmailRegex()}.
     *
     * For personal preference, I include a test name as the first parameter
     * which describes what that test should be testing - this is visible in
     * IntelliJ when running the tests (see above note if not working).
     */
    @ParameterizedTest
    @MethodSource
    public void testEmailRegex(String test, String input, boolean success) {
        test(input, Regex.EMAIL, success);
    }

    /**
     * This is the factory method providing test cases for the parameterized
     * test above - note that it is static, takes no arguments, and has the same
     * name as the test. The {@link Arguments} object contains the arguments for
     * each test to be passed to the function above.
     */
    public static Stream<Arguments> testEmailRegex() {
        return Stream.of(
                Arguments.of("Alphanumeric", "thelegend27@gmail.com", true),
                Arguments.of("Numeric", "27@gmail.com", true),
                Arguments.of("UF Domain", "otherdomain@ufl.edu", true),
                Arguments.of("Missing Domain", "hello@.com", true),
                Arguments.of("Upper Case", "ADDRESS@UFL.edu", true),
                Arguments.of("Special Chars", "other.domain_-@ufl-fl.edu", true),
                Arguments.of("Only Special Chars", "._-@ufl-fl.edu", true),
                Arguments.of("Alphanumeric Domain", "otherdomain@ufl42.edu", true),
                Arguments.of("Numeric Domain", "otherdomain@42.edu", true),
                Arguments.of("Double Dot Address", "other.domain.Bob@ufl.edu", true),
                Arguments.of("Missing Domain Dot", "missingdot@gmailcom", false),
                Arguments.of("Invalid Symbols", "symbols!#$%^&*()~@gmail.com", false),
                Arguments.of("Missing @", "hellogmail.com", false),
                Arguments.of("Double @", "hello@gmail.com@hotmail.com", false),
                Arguments.of("Long After Dot", "hello@gmail.comm", false),
                Arguments.of("Short After Dot", "hello@gmail.i", false),
                Arguments.of("Empty Address", "@gmail.com", false),
                Arguments.of("Long After Dot", "hello@gmail.comm", false),
                Arguments.of("Symbol Domain", "hello@gm@il~!#$%^&*().com", false),
                Arguments.of("Upper Case .Com", "otherdomain@ufl.EDU", false),
                Arguments.of("Symbol .com", "otherdomain@ufl.@#$", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    public void testOddStringsRegex(String test, String input, boolean success) {
        test(input, Regex.ODD_STRINGS, success);
    }

    public static Stream<Arguments> testOddStringsRegex() {
        return Stream.of(
                Arguments.of("10 Characters", "automobile", false),
                Arguments.of("14 Characters", "i<3pancakes10!", false),
                Arguments.of("20 Characters", "hyperconsciousnesses", false),
                Arguments.of("20 Random Characters", " Y!@R 0nsc1ous^esses", false),
                Arguments.of("19 Random Characters", " Y!@R 0nsc1ous^esse", true),
                Arguments.of("11 Spaces", "           ", true),
                Arguments.of("19 Characters", "1234567890123456789", true),
                Arguments.of("17 Characters", "12345678901234567", true),
                Arguments.of("15 Characters", "123456789012345", true),
                Arguments.of("Alphabetic", "abcdefghijklm", true),
                Arguments.of("Numeric", "123456789012345", true),
                Arguments.of("Alphanumeric", "1234567abcdef", true),
                Arguments.of("Special Chars", "!@#$%^&*()!@!", true),
                Arguments.of("6 Characters", "6chars", false),
                Arguments.of("13 Characters", "i<3pancakes9!", true),
                Arguments.of("8 Characters", "Absolute", false),
                Arguments.of("22 Characters", "spectrophotometrically", false),
                Arguments.of("10 Spaces", "          ", false),
                Arguments.of("Empty", "", false),
                Arguments.of("21 Characters", "anthropomorphizations", false),
                Arguments.of("9 Characters", "aardvarks", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    public void testIntegerListRegex(String test, String input, boolean success) {
        test(input, Regex.INTEGER_LIST, success);
    }

    public static Stream<Arguments> testIntegerListRegex() {
        return Stream.of(
                Arguments.of("Single Element", "[1]", true),
                Arguments.of("Multiple Elements", "[1,2,3]", true),
                Arguments.of("Empty", "[]", true),
                Arguments.of("Big List", "[1,2,3,4,5,6,7,8,9]", true),
                Arguments.of("No Whitespace", "[1,2,3,4]", true),
                Arguments.of("Mismatched Whitespace", "[1, 2,3,4, 5,6, 7,8,9]", true),
                Arguments.of("Big numbers", "[123445578, 12314253463547373]", true),
                Arguments.of("Multiple Spaces", "[1,    3,     7]", true),
                Arguments.of("Leading Space", "[ 1, 2, 3]", true),
                Arguments.of("Trailing Spaces", "[1, 2, 3 ]", true),
                Arguments.of("Missing Brackets", "1,2,3", false),
                Arguments.of("Missing End Bracket", "[1,2,3", false),
                Arguments.of("Missing Front Bracket", "1,2,3]", false),
                Arguments.of("Missing Commas", "[1 2 3]", false),
                Arguments.of("Trailing Comma", "[1, 2, 3,]", false),
                Arguments.of("Leading Comma", "[,1, 2, 3]", false),
                Arguments.of("Just Commas", "[,,,]", false),
                Arguments.of("Double Commas", "[1,,3]", false),
                Arguments.of("Negatives", "[1,-3]", false),
                Arguments.of("Just Commas and Whitespace", "[, , ,, ,,]", false),
                Arguments.of("Characters", "[a, b, c]", false),
                Arguments.of("Symbols", "[!, @, $]", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    public void testDecimalRegex(String test, String input, boolean success) {
        test(input, Regex.DECIMAL, success);
    }

    public static Stream<Arguments> testDecimalRegex(){
        return Stream.of(
                Arguments.of("Decimal", "1234567890.123456789", true),
                Arguments.of("Negative Decimal", "-1234567890.123456789", true),
                Arguments.of("Small Decimal", "1.9", true),
                Arguments.of("Negative Small Decimal", "-1.9", true),
                Arguments.of("Zero in Front", "0.9", true),
                Arguments.of("Negative Zero in Front", "-0.9", true),
                Arguments.of("Nothing in front", ".5", false),
                Arguments.of("Nothing in Back", "1", false),
                Arguments.of("Random Negative", "1-.5", false),
                Arguments.of("Random Negative2", "1.-5", false),
                Arguments.of("Random Negative3", "1.5-", false),
                Arguments.of("Leading 0", "01.5", false),
                Arguments.of("Multiple Leading 0", "0001.5", false),
                Arguments.of("Positive Sign", "+1.5", false),
                Arguments.of("Positive Sign Only", "+", false),
                Arguments.of("Negative Sign Only", "-", false),
                Arguments.of("Other Symbol", "^1.9", false),
                Arguments.of("Chars", "abcABC", false),
                Arguments.of("Invalid Decimal", "1:9", false),
                Arguments.of("Invalid Decimal2", "1,9", false),
                Arguments.of("Unicode", "႑", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    public void testStringRegex(String test, String input, boolean success) {
        test(input, Regex.STRING, success);
    }

    public static Stream<Arguments> testStringRegex() {
        return Stream.of(
                Arguments.of("Empty", "\"\"", true),
                Arguments.of("Basic", "\"Hello World!\"", true),
                Arguments.of("Escape", "\"1\t2\"", true),
                Arguments.of("Escape2", "\"1\b2\"", true),
                Arguments.of("Escape3", "\"1\n2\"", true),
                Arguments.of("Escape4", "\"1\r2\"", true),
                Arguments.of("Escape5", "\"1\'2\"", true),
                Arguments.of("Escape6", "\"1\"2\"", true),
                Arguments.of("Escape7", "\"1\\\\2\"", true),
                Arguments.of("Symbols", "\"!@#$$%^&^&*\"", true),
                Arguments.of("Whitespace", "\"       \"", true),
                Arguments.of("Unterminated End", "\"Nope", false),
                Arguments.of("Unstarded", "Nope\"", false),
                Arguments.of("Invalid Escape", "\"invalid\\escape\"", false),
                Arguments.of("Unicode Escape", "\"a\\u0000b\\u12ABc\"", false),
                Arguments.of("Trailing Chars", "\"abc\"abc", false),
                Arguments.of("Leading Chars", "abc\"abc\"", false),
                Arguments.of("True Empty", "", false),
                Arguments.of("Unterminated Empty", "\"", false)
        );
    }

    /**
     * Asserts that the input matches the given pattern. This method doesn't do
     * much now, but you will see this concept in future assignments.
     */
    private static void test(String input, Pattern pattern, boolean success) {
        Assertions.assertEquals(success, pattern.matcher(input).matches());
    }

}
