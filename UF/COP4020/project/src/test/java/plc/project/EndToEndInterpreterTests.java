package plc.project;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.stream.Stream;
import java.util.function.Function;

final class EndToEndInterpreterTests {

    @ParameterizedTest
    @MethodSource
    void testSource(String test, String input, Object expected) {
        test(input, expected, new Scope(null), Parser::parseSource);
    }

    private static Stream<Arguments> testSource() {
        return Stream.of(
                // DEF main() DO RETURN 0; END
                Arguments.of("Main",
                        "DEF main() DO RETURN 0; END"
                        , BigInteger.ZERO
                ),
                // LET x: Integer = 1; LET y: Integer = 10; DEF main() DO x + y; END
                Arguments.of("Fields & No Return",
                        "LET x: Integer = 1; LET y: Integer = 10; DEF main() DO x + y; END",
                        Environment.NIL.getValue()
                )
        );
    }

    @ParameterizedTest
    @MethodSource
    void testField(String test, String input, Object expected, String variableName) {
        Scope scope = test(input, Environment.NIL.getValue(), new Scope(null), Parser::parseField);
        Assertions.assertEquals(expected, scope.lookupVariable(variableName).getValue().getValue());
    }

    private static Stream<Arguments> testField() {
        return Stream.of(
                // LET name: Integer;
                Arguments.of("Field",
                        "LET name: Integer;",
                        Environment.NIL.getValue(),
                        "name"
                ),
                // LET name: Integer = 1;
                Arguments.of("Initialization",
                        "LET name: Integer = 1;",
                        BigInteger.ONE,
                        "name"
                )
        );
    }

    @ParameterizedTest
    @MethodSource
    void testMethod(String test, String input, List<Environment.PlcObject> args, Object expected, String functionName) {
        Scope scope = test(input, Environment.NIL.getValue(), new Scope(null), Parser::parseMethod);
        Assertions.assertEquals(expected, scope.lookupFunction(functionName, args.size()).invoke(args).getValue());
    }

    private static Stream<Arguments> testMethod() {
        return Stream.of(
                // DEF main(): Integer DO RETURN 0; END
                Arguments.of("Main",
                        "DEF main(): Integer DO RETURN 0; END",
                        List.of(),
                        BigInteger.ZERO,
                        "main"
                ),
                // DEF square(x: Integer): Integer DO RETURN x * x; END
                Arguments.of("Arguments",
                        "DEF square(x: Integer): Integer DO RETURN x * x; END",
                        List.of(Environment.create(BigInteger.TEN)),
                        BigInteger.valueOf(100),
                        "square"
                )
        );
    }

    @Test
    void testExpressionStatement() {
        // print("Hello, World!");
        PrintStream sysout = System.out;
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        System.setOut(new PrintStream(out));
        try {
            test("print(\"Hello, World!\");", Environment.NIL.getValue(), new Scope(null), Parser::parseStatement);
            Assertions.assertEquals("Hello, World!" + System.lineSeparator(), out.toString());
        } finally {
            System.setOut(sysout);
        }
    }

    @ParameterizedTest
    @MethodSource
    void testDeclarationStatement(String test, String input, Object expected, String variableName) {
        Scope scope = test(input, Environment.NIL.getValue(), new Scope(null), Parser::parseStatement);
        Assertions.assertEquals(expected, scope.lookupVariable(variableName).getValue().getValue());
    }

    private static Stream<Arguments> testDeclarationStatement() {
        return Stream.of(
                // LET name;
                Arguments.of("Declaration",
                        "LET name;",
                        Environment.NIL.getValue(),
                        "name"
                ),
                // LET name = 1;
                Arguments.of("Initialization",
                        "LET name = 1;",
                        BigInteger.ONE,
                        "name"
                )
        );
    }

    @Test
    void testVariableAssignmentStatement() {
        // variable = 1;
        Scope scope = new Scope(null);
        scope.defineVariable("variable", false, Environment.create("variable"));
        test("variable = 1;", Environment.NIL.getValue(), scope, Parser::parseStatement);
        Assertions.assertEquals(BigInteger.ONE, scope.lookupVariable("variable").getValue().getValue());
    }

    @ParameterizedTest
    @MethodSource
    void testIfStatement(String test, String input, Object expected) {
        Scope scope = new Scope(null);
        scope.defineVariable("num", false, Environment.NIL);
        test(input, Environment.NIL.getValue(), scope, Parser::parseStatement);
        Assertions.assertEquals(expected, scope.lookupVariable("num").getValue().getValue());
    }

    private static Stream<Arguments> testIfStatement() {
        return Stream.of(
                // IF TRUE DO num = 1; END
                Arguments.of("True Condition",
                        "IF TRUE DO num = 1; END",
                        BigInteger.ONE
                ),
                // IF FALSE DO ELSE num = 10; END
                Arguments.of("False Condition",
                        "IF FALSE DO ELSE num = 10; END",
                        BigInteger.TEN
                )
        );
    }

    @Test
    void testForStatement() {
        // FOR (num = 0; num < 5; num = num + 1) sum = sum + num; END
        Scope scope = new Scope(null);
        scope.defineVariable("sum", false, Environment.create(BigInteger.ZERO));
        scope.defineVariable("num", false, Environment.NIL);

        String input = new String("FOR (num = 0; num < 5; num = num + 1) sum = sum + num; END");

        test(input, Environment.NIL.getValue(), scope, Parser::parseStatement);
        Assertions.assertEquals(BigInteger.TEN, scope.lookupVariable("sum").getValue().getValue());
        Assertions.assertEquals(BigInteger.valueOf(5), scope.lookupVariable("num").getValue().getValue());
    }

    @Test
    void testWhileStatement() {
        // WHILE num < 10 DO num = num + 1; END
        Scope scope = new Scope(null);
        scope.defineVariable("num", false, Environment.create(BigInteger.ZERO));
        test("WHILE num < 10 DO num = num + 1; END",Environment.NIL.getValue(), scope, Parser::parseStatement);
        Assertions.assertEquals(BigInteger.TEN, scope.lookupVariable("num").getValue().getValue());
    }

    @ParameterizedTest
    @MethodSource
    void testLiteralExpression(String test, String input, Object expected) {
        test(input, expected, new Scope(null), Parser::parseExpression);
    }

    private static Stream<Arguments> testLiteralExpression() {
        return Stream.of(
                // NIL
                Arguments.of("Nil", "NIL", Environment.NIL.getValue()), //remember, special case
                // TRUE
                Arguments.of("Boolean", "TRUE", true),
                // 1
                Arguments.of("Integer", "1", BigInteger.ONE),
                // 1.0
                Arguments.of("Decimal", "1.0", new BigDecimal("1.0")),
                // 'c'
                Arguments.of("Character", "'c'", 'c'),
                // "string"
                Arguments.of("String", "\"string\"", "string")
        );
    }

    @ParameterizedTest
    @MethodSource
    void testGroupExpression(String test, String input, Object expected) {
        test(input, expected, new Scope(null), Parser::parseExpression);
    }

    private static Stream<Arguments> testGroupExpression() {
        return Stream.of(
                // (1)
                Arguments.of("Literal",
                        "(1)",
                        BigInteger.ONE
                ),
                // (1 + 10)
                Arguments.of("Binary",
                        "(1 + 10)",
                        BigInteger.valueOf(11)
                )
        );
    }

    @ParameterizedTest
    @MethodSource
    void testBinaryExpression(String test, String input, Object expected) {
        test(input, expected, new Scope(null), Parser::parseExpression);
    }

    private static Stream<Arguments> testBinaryExpression() {
        return Stream.of(
                // TRUE && FALSE
                Arguments.of("And",
                        "TRUE && FALSE",
                        false
                ),
                // TRUE || undefined
                Arguments.of("Or (Short Circuit)",
                        "TRUE || undefined",
                        true
                ),
                // 1 < 10
                Arguments.of("Less Than",
                        "1 < 10",
                        true
                ),
                // 1 == 10
                Arguments.of("Equal",
                        "1 == 10",
                        false
                ),
                // "a" + "b"
                Arguments.of("Concatenation",
                        "\"a\" + \"b\"",
                        "ab"
                ),
                // 1 + 10
                Arguments.of("Addition",
                        "1 + 10",
                        BigInteger.valueOf(11)
                ),
                // 1.2 / 3.4
                Arguments.of("Division",
                        "1.2 / 3.4",
                        new BigDecimal("0.4")
                )
        );
    }

    @ParameterizedTest
    @MethodSource
    void testAccessExpression(String test, String input, Object expected) {
        Scope scope = new Scope(null);
        scope.defineVariable("variable", false, Environment.create("variable"));
        test(input, expected, scope, Parser::parseExpression);
    }

    private static Stream<Arguments> testAccessExpression() {
        return Stream.of(
                // variable
                Arguments.of("Variable",
                        "variable",
                        "variable"
                )
        );
    }

    @ParameterizedTest
    @MethodSource
    void testFunctionExpression(String test, String input, Object expected) {
        Scope scope = new Scope(null);
        scope.defineFunction("function", 0, args -> Environment.create("function"));
        test(input, expected, scope, Parser::parseExpression);
    }

    private static Stream<Arguments> testFunctionExpression() {
        return Stream.of(
                // function()
                Arguments.of("Function",
                        "function()",
                        "function"
                ),
                // print("Hello, World!")
                Arguments.of("Print",
                        "print(\"Hello, World!\")",
                        Environment.NIL.getValue()
                )
        );
    }

    private static <T extends Ast> Scope test(String input, Object expected, Scope scope, Function<Parser, T> function) {
        Lexer lexer = new Lexer(input);
        Parser parser = new Parser(lexer.lex());

        Ast ast = function.apply(parser);

        Interpreter interpreter = new Interpreter(scope);
        if (expected != null) {
            Assertions.assertEquals(expected, interpreter.visit(ast).getValue());
        } else {
            Assertions.assertThrows(RuntimeException.class, () -> interpreter.visit(ast));
        }
        return interpreter.getScope();
    }

}
