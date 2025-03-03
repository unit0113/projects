package plc.project;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class LexerTests {

    @ParameterizedTest
    @MethodSource
    void testIdentifier(String test, String input, boolean success) {
        test(input, Token.Type.IDENTIFIER, success);
    }

    private static Stream<Arguments> testIdentifier() {
        return Stream.of(
                Arguments.of("Alphabetic", "getName", true),
                Arguments.of("Alphabetic2", "abc", true),
                Arguments.of("Alphanumeric", "thelegend27", true),
                Arguments.of("Single", "a", true),
                Arguments.of("Hyphen Underscore", "t-h_e", true),
                Arguments.of("Leading Hyphen", "-five", false),
                Arguments.of("Leading Digit", "1fish2fish3fishbluefish", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testInteger(String test, String input, boolean success) {
        test(input, Token.Type.INTEGER, success);
    }

    private static Stream<Arguments> testInteger() {
        return Stream.of(
                Arguments.of("Single Digit", "1", true),
                Arguments.of("Single Zero", "0", true),
                Arguments.of("Multiple Digits", "12345", true),
                Arguments.of("Negative", "-1", true),
                Arguments.of("Positive", "+1", true),
                Arguments.of("Leading Zero", "01", false),
                Arguments.of("Multiple Leading Zero", "001", false),
                Arguments.of("Negative Leading Zero", "-01", false),
                Arguments.of("Positive Leading Zero", "+01", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testDecimal(String test, String input, boolean success) {
        test(input, Token.Type.DECIMAL, success);
    }

    private static Stream<Arguments> testDecimal() {
        return Stream.of(
                Arguments.of("Multiple Digits", "123.456", true),
                Arguments.of("Negative Decimal", "-1.0", true),
                Arguments.of("Leading 0", "0.5", true),
                Arguments.of("Negative", "-1.5", true),
                Arguments.of("Positive", "+1.9", true),
                Arguments.of("Negative Leading Zero", "-0.1", true),
                Arguments.of("Positive Leading Zero", "+0.1", true),
                Arguments.of("Trailing Decimal", "1.", false),
                Arguments.of("Leading Decimal", ".5", false),
                Arguments.of("Operator", "5.toString()", false),
                Arguments.of("Double decimal", "5..45", false),
                Arguments.of("Double decimal2", "5.6.45", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testCharacter(String test, String input, boolean success) {
        test(input, Token.Type.CHARACTER, success);
    }

    private static Stream<Arguments> testCharacter() {
        return Stream.of(
                Arguments.of("Alphabetic", "\'c\'", true),
                Arguments.of("Newline Escape", "\'\\n\'", true),
                Arguments.of("Empty", "\'\'", false),
                Arguments.of("Multiple", "\'abc\'", false),
                Arguments.of("Unterminated", "\'a", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testString(String test, String input, boolean success) {
        test(input, Token.Type.STRING, success);
    }

    private static Stream<Arguments> testString() {
        return Stream.of(
                Arguments.of("Empty", "\"\"", true),
                Arguments.of("Alphabetic", "\"abc\"", true),
                Arguments.of("Newline Escape", "\"Hello,\\nWorld\"", true),
                Arguments.of("Symbols", "\"!@#$%^&*()\"", true),
                Arguments.of("Unterminated", "\"unterminated\n", false),
                Arguments.of("Invalid Escape", "\"invalid\\escape\"", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testOperator(String test, String input, boolean success) {
        //this test requires our lex() method, since that's where whitespace is handled.
        test(input, Arrays.asList(new Token(Token.Type.OPERATOR, input, 0)), success);
    }

    private static Stream<Arguments> testOperator() {
        return Stream.of(
                Arguments.of("Character", "(", true),
                Arguments.of("Comparison", "!=", true),
                Arguments.of("Comparison2", "<=", true),
                Arguments.of("Comparison3", ">=", true),
                Arguments.of("Comparison4", "==", true),
                Arguments.of("Comparison5", "<", true),
                Arguments.of("Comparison6", ">", true),
                Arguments.of("And", "&&", true),
                Arguments.of("Or", "||", true),
                Arguments.of("Space", " ", false),
                Arguments.of("Tab", "\t", false)
        );
    }

    @ParameterizedTest
    @MethodSource
    void testExamples(String test, String input, List<Token> expected) {
        test(input, expected, true);
    }

    private static Stream<Arguments> testExamples() {
        return Stream.of(
                Arguments.of("Example 1", "LET x = 5;", Arrays.asList(
                        new Token(Token.Type.IDENTIFIER, "LET", 0),
                        new Token(Token.Type.IDENTIFIER, "x", 4),
                        new Token(Token.Type.OPERATOR, "=", 6),
                        new Token(Token.Type.INTEGER, "5", 8),
                        new Token(Token.Type.OPERATOR, ";", 9)
                )),
                Arguments.of("Example 2", "print(\"Hello, World!\");", Arrays.asList(
                        new Token(Token.Type.IDENTIFIER, "print", 0),
                        new Token(Token.Type.OPERATOR, "(", 5),
                        new Token(Token.Type.STRING, "\"Hello, World!\"", 6),
                        new Token(Token.Type.OPERATOR, ")", 21),
                        new Token(Token.Type.OPERATOR, ";", 22)
                )),
                Arguments.of("Extra Whitespace", "LET  x = 5;", Arrays.asList(
                        new Token(Token.Type.IDENTIFIER, "LET", 0),
                        new Token(Token.Type.IDENTIFIER, "x", 5),
                        new Token(Token.Type.OPERATOR, "=", 7),
                        new Token(Token.Type.INTEGER, "5", 9),
                        new Token(Token.Type.OPERATOR, ";", 10)
                )),
                Arguments.of("FizzBuzz", "LET i = 1;\n" +
                        "WHILE i != 100 DO\n" +
                        "    IF rem(i, 3) == 0 && rem(i, 5) == 0 DO\n" +
                        "        print(\"FizzBuzz\");\n" +
                        "    ELSE IF rem(i, 3) == 0 DO\n" +
                        "        print(\"Fizz\");\n" +
                        "    ELSE IF rem(i, 5) == 0 DO\n" +
                        "        print(\"Buzz\");\n" +
                        "    ELSE\n" +
                        "        print(i);\n" +
                        "    END END END\n" +
                        "    i = i + 1;\n" +
                        "END", Arrays.asList(
                        new Token(Token.Type.IDENTIFIER, "LET", 0),
                        new Token(Token.Type.IDENTIFIER, "i", 4),
                        new Token(Token.Type.OPERATOR, "=", 6),
                        new Token(Token.Type.INTEGER, "1", 8),
                        new Token(Token.Type.OPERATOR, ";", 9),
                        new Token(Token.Type.IDENTIFIER, "WHILE", 11),
                        new Token(Token.Type.IDENTIFIER, "i", 17),
                        new Token(Token.Type.OPERATOR, "!=", 19),
                        new Token(Token.Type.INTEGER, "100", 22),
                        new Token(Token.Type.IDENTIFIER, "DO", 26),
                        new Token(Token.Type.IDENTIFIER, "IF", 33),
                        new Token(Token.Type.IDENTIFIER, "rem", 36),
                        new Token(Token.Type.OPERATOR, "(", 39),
                        new Token(Token.Type.IDENTIFIER, "i", 40),
                        new Token(Token.Type.OPERATOR, ",", 41),
                        new Token(Token.Type.INTEGER, "3", 43),
                        new Token(Token.Type.OPERATOR, ")", 44),
                        new Token(Token.Type.OPERATOR, "==", 46),
                        new Token(Token.Type.INTEGER, "0", 49),
                        new Token(Token.Type.OPERATOR, "&&", 51),
                        new Token(Token.Type.IDENTIFIER, "rem", 54),
                        new Token(Token.Type.OPERATOR, "(", 57),
                        new Token(Token.Type.IDENTIFIER, "i", 58),
                        new Token(Token.Type.OPERATOR, ",", 59),
                        new Token(Token.Type.INTEGER, "5", 61),
                        new Token(Token.Type.OPERATOR, ")", 62),
                        new Token(Token.Type.OPERATOR, "==", 64),
                        new Token(Token.Type.INTEGER, "0", 67),
                        new Token(Token.Type.IDENTIFIER, "DO", 69),
                        new Token(Token.Type.IDENTIFIER, "print", 80),
                        new Token(Token.Type.OPERATOR, "(", 85),
                        new Token(Token.Type.STRING, "\"FizzBuzz\"", 86),
                        new Token(Token.Type.OPERATOR, ")", 96),
                        new Token(Token.Type.OPERATOR, ";", 97),
                        new Token(Token.Type.IDENTIFIER, "ELSE", 103),
                        new Token(Token.Type.IDENTIFIER, "IF", 108),
                        new Token(Token.Type.IDENTIFIER, "rem", 111),
                        new Token(Token.Type.OPERATOR, "(", 114),
                        new Token(Token.Type.IDENTIFIER, "i", 115),
                        new Token(Token.Type.OPERATOR, ",", 116),
                        new Token(Token.Type.INTEGER, "3", 118),
                        new Token(Token.Type.OPERATOR, ")", 119),
                        new Token(Token.Type.OPERATOR, "==", 121),
                        new Token(Token.Type.INTEGER, "0", 124),
                        new Token(Token.Type.IDENTIFIER, "DO", 126),
                        new Token(Token.Type.IDENTIFIER, "print", 137),
                        new Token(Token.Type.OPERATOR, "(", 142),
                        new Token(Token.Type.STRING, "\"Fizz\"", 143),
                        new Token(Token.Type.OPERATOR, ")", 149),
                        new Token(Token.Type.OPERATOR, ";", 150),
                        new Token(Token.Type.IDENTIFIER, "ELSE", 156),
                        new Token(Token.Type.IDENTIFIER, "IF", 161),
                        new Token(Token.Type.IDENTIFIER, "rem", 164),
                        new Token(Token.Type.OPERATOR, "(", 167),
                        new Token(Token.Type.IDENTIFIER, "i", 168),
                        new Token(Token.Type.OPERATOR, ",", 169),
                        new Token(Token.Type.INTEGER, "5", 171),
                        new Token(Token.Type.OPERATOR, ")", 172),
                        new Token(Token.Type.OPERATOR, "==", 174),
                        new Token(Token.Type.INTEGER, "0", 177),
                        new Token(Token.Type.IDENTIFIER, "DO", 179),
                        new Token(Token.Type.IDENTIFIER, "print", 190),
                        new Token(Token.Type.OPERATOR, "(", 195),
                        new Token(Token.Type.STRING, "\"Buzz\"", 196),
                        new Token(Token.Type.OPERATOR, ")", 202),
                        new Token(Token.Type.OPERATOR, ";", 203),
                        new Token(Token.Type.IDENTIFIER, "ELSE", 209),
                        new Token(Token.Type.IDENTIFIER, "print", 222),
                        new Token(Token.Type.OPERATOR, "(", 227),
                        new Token(Token.Type.IDENTIFIER, "i", 228),
                        new Token(Token.Type.OPERATOR, ")", 229),
                        new Token(Token.Type.OPERATOR, ";", 230),
                        new Token(Token.Type.IDENTIFIER, "END", 236),
                        new Token(Token.Type.IDENTIFIER, "END", 240),
                        new Token(Token.Type.IDENTIFIER, "END", 244),
                        new Token(Token.Type.IDENTIFIER, "i", 252),
                        new Token(Token.Type.OPERATOR, "=", 254),
                        new Token(Token.Type.IDENTIFIER, "i", 256),
                        new Token(Token.Type.OPERATOR, "+", 258),
                        new Token(Token.Type.INTEGER, "1", 260),
                        new Token(Token.Type.OPERATOR, ";", 261),
                        new Token(Token.Type.IDENTIFIER, "END", 263)
                ))
        );
    }

    @Test
    void testException() {
        ParseException exception = Assertions.assertThrows(ParseException.class,
                () -> new Lexer("\"unterminated").lex());
        Assertions.assertEquals(13, exception.getIndex());
    }

    /**
     * Tests that lexing the input through {@link Lexer#lexToken()} produces a
     * single token with the expected type and literal matching the input.
     */
    private static void test(String input, Token.Type expected, boolean success) {
        try {
            if (success) {
                Assertions.assertEquals(new Token(expected, input, 0), new Lexer(input).lexToken());
            } else {
                Assertions.assertNotEquals(new Token(expected, input, 0), new Lexer(input).lexToken());
            }
        } catch (ParseException e) {
            Assertions.assertFalse(success, e.getMessage());
        }
    }

    /**
     * Tests that lexing the input through {@link Lexer#lex()} matches the
     * expected token list.
     */
    private static void test(String input, List<Token> expected, boolean success) {
        try {
            if (success) {
                Assertions.assertEquals(expected, new Lexer(input).lex());
            } else {
                Assertions.assertNotEquals(expected, new Lexer(input).lex());
            }
        } catch (ParseException e) {
            Assertions.assertFalse(success, e.getMessage());
        }
    }

}
