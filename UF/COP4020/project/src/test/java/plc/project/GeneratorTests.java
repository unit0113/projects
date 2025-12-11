package plc.project;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tags;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import javax.swing.text.html.Option;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class GeneratorTests {

    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testSource(String test, Ast.Source ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testSource() {
        return Stream.of(
                Arguments.of("Hello, World!",
                        // FUN main(): Integer DO
                        //     print("Hello, World!");
                        //     RETURN 0;
                        // END
                        new Ast.Source(
                                Arrays.asList(),
                                Arrays.asList(init(new Ast.Method("main", Arrays.asList(), Arrays.asList(), Optional.of("Integer"), Arrays.asList(
                                        new Ast.Statement.Expression(init(new Ast.Expression.Function(Optional.empty(), "print", Arrays.asList(
                                                init(new Ast.Expression.Literal("Hello, World!"), ast -> ast.setType(Environment.Type.STRING))
                                        )), ast -> ast.setFunction(new Environment.Function("print", "System.out.println", Arrays.asList(Environment.Type.ANY), Environment.Type.NIL, args -> Environment.NIL)))),
                                        new Ast.Statement.Return(init(new Ast.Expression.Literal(BigInteger.ZERO), ast -> ast.setType(Environment.Type.INTEGER)))
                                )), ast -> ast.setFunction(new Environment.Function("main", "main", Arrays.asList(), Environment.Type.INTEGER, args -> Environment.NIL))))
                        ),
                        String.join(System.lineSeparator(),
                                "public class Main {",
                                "",
                                "    public static void main(String[] args) {",
                                "        System.exit(new Main().main());",
                                "    }",
                                "",
                                "    int main() {",
                                "        System.out.println(\"Hello, World!\");",
                                "        return 0;",
                                "    }",
                                "",
                                "}"
                        )
                )
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testDeclarationStatement(String test, Ast.Statement.Declaration ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testDeclarationStatement() {
        return Stream.of(
                Arguments.of("Declaration",
                        // LET name: Integer;
                        init(new Ast.Statement.Declaration("name", Optional.of("Integer"), Optional.empty()), ast -> ast.setVariable(new Environment.Variable("name", "name", Environment.Type.INTEGER, true, Environment.NIL))),
                        "int name;"
                ),
                Arguments.of("Initialization",
                        // LET name = 1.0;
                        init(new Ast.Statement.Declaration("name", Optional.empty(), Optional.of(
                                init(new Ast.Expression.Literal(new BigDecimal("1.0")),ast -> ast.setType(Environment.Type.DECIMAL))
                        )), ast -> ast.setVariable(new Environment.Variable("name", "name", Environment.Type.DECIMAL, true, Environment.NIL))),
                        "double name = 1.0;"
                )
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testIfStatement(String test, Ast.Statement.If ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testIfStatement() {
        return Stream.of(
                Arguments.of("If",
                        // IF expr DO
                        //     stmt;
                        // END
                        new Ast.Statement.If(
                                init(new Ast.Expression.Access(Optional.empty(), "expr"), ast -> ast.setVariable(new Environment.Variable("expr", "expr", Environment.Type.BOOLEAN, true, Environment.NIL))),
                                Arrays.asList(new Ast.Statement.Expression(init(new Ast.Expression.Access(Optional.empty(), "stmt"), ast -> ast.setVariable(new Environment.Variable("stmt", "stmt", Environment.Type.NIL, true, Environment.NIL))))),
                                Arrays.asList()
                        ),
                        String.join(System.lineSeparator(),
                                "if (expr) {",
                                "    stmt;",
                                "}"
                        )
                ),
                Arguments.of("Else",
                        // IF expr DO
                        //     stmt1;
                        // ELSE
                        //     stmt2;
                        // END
                        new Ast.Statement.If(
                                init(new Ast.Expression.Access(Optional.empty(), "expr"), ast -> ast.setVariable(new Environment.Variable("expr", "expr", Environment.Type.BOOLEAN, true, Environment.NIL))),
                                Arrays.asList(new Ast.Statement.Expression(init(new Ast.Expression.Access(Optional.empty(), "stmt1"), ast -> ast.setVariable(new Environment.Variable("stmt1", "stmt1", Environment.Type.NIL, true, Environment.NIL))))),
                                Arrays.asList(new Ast.Statement.Expression(init(new Ast.Expression.Access(Optional.empty(), "stmt2"), ast -> ast.setVariable(new Environment.Variable("stmt2", "stmt2", Environment.Type.NIL, true, Environment.NIL)))))
                        ),
                        String.join(System.lineSeparator(),
                                "if (expr) {",
                                "    stmt1;",
                                "} else {",
                                "    stmt2;",
                                "}"
                        )
                )
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testForStatement(String test, Ast.Statement.For ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testForStatement() {
        return Stream.of(
                Arguments.of("For",
                            // for (num = 0; num < 5; num = num + 1)
                            //     print(num);
                            // END
                        new Ast.Statement.For(
                                new Ast.Statement.Assignment(
                                        init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                        init(new Ast.Expression.Literal(BigInteger.valueOf(0)),
                                                ast -> ast.setType(Environment.Type.INTEGER))),
                                init(new Ast.Expression.Binary("<",
                                                init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                        ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                                init(new Ast.Expression.Literal(BigInteger.valueOf(5)),
                                                        ast -> ast.setType(Environment.Type.INTEGER))),
                                        ast -> ast.setType(Environment.Type.BOOLEAN)),
                                new Ast.Statement.Assignment(
                                        init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                        init(new Ast.Expression.Binary("+",
                                                        init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                                ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                                        init(new Ast.Expression.Literal(BigInteger.valueOf(1)),
                                                                ast -> ast.setType(Environment.Type.INTEGER))),
                                                ast -> ast.setType(Environment.Type.INTEGER))),
                                Arrays.asList(
                                        new Ast.Statement.Expression(
                                                init(new Ast.Expression.Function(Optional.empty(),"print", Arrays.asList(
                                                        init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                                ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))))),
                                                        ast -> ast.setFunction(new Environment.Function("print", "System.out.println", Arrays.asList(Environment.Type.ANY), Environment.Type.NIL, args -> Environment.NIL))))
                                )
                        ),
                        String.join(System.lineSeparator(),
                                "for ( num = 0; num < 5; num = num + 1 ) {",
                                "    System.out.println(num);",
                                "}"
                        )
                ),
                Arguments.of("Missing Signature",
                        // for (; num < 5;)
                        //     print(num);
                        // END
                        new Ast.Statement.For(
                                null,
                                init(new Ast.Expression.Binary("<",
                                                init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                        ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                                init(new Ast.Expression.Literal(BigInteger.valueOf(5)),
                                                        ast -> ast.setType(Environment.Type.INTEGER))),
                                        ast -> ast.setType(Environment.Type.BOOLEAN)),
                                null,
                                Arrays.asList(
                                        new Ast.Statement.Expression(
                                                init(new Ast.Expression.Function(Optional.empty(),"print", Arrays.asList(
                                                                init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                                        ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))))),
                                                        ast -> ast.setFunction(new Environment.Function("print", "System.out.println", Arrays.asList(Environment.Type.ANY), Environment.Type.NIL, args -> Environment.NIL)))),
                                        new Ast.Statement.Assignment(
                                                init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                        ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                                init(new Ast.Expression.Binary("+",
                                                                init(new Ast.Expression.Access(Optional.empty(), "num"),
                                                                        ast -> ast.setVariable(new Environment.Variable("num", "num", Environment.Type.INTEGER, false, Environment.NIL))),
                                                                init(new Ast.Expression.Literal(BigInteger.valueOf(1)),
                                                                        ast -> ast.setType(Environment.Type.INTEGER))),
                                                        ast -> ast.setType(Environment.Type.INTEGER)
                                                )
                                        )
                                )
                        ),
                        String.join(System.lineSeparator(),
                                "for ( ; num < 5; ) {",
                                "    System.out.println(num);",
                                "    num = num + 1;",
                                "}"
                        )
                )
        );
    }

    // for (; num < 5;)
    //     print(num);
    //     num = num + 1;
    // END

//                "for ( num = 0; num < 5; num = num + 1 ) {",
//        "    System.out.println(num);",
//                "END"


    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testBinaryExpression(String test, Ast.Expression.Binary ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testBinaryExpression() {
        return Stream.of(
                Arguments.of("And",
                        // TRUE && FALSE
                        init(new Ast.Expression.Binary("&&",
                                init(new Ast.Expression.Literal(true), ast -> ast.setType(Environment.Type.BOOLEAN)),
                                init(new Ast.Expression.Literal(false), ast -> ast.setType(Environment.Type.BOOLEAN))
                        ), ast -> ast.setType(Environment.Type.BOOLEAN)),
                        "true && false"
                ),
                Arguments.of("Concatenation",
                        // "Ben" + 10
                        init(new Ast.Expression.Binary("+",
                                init(new Ast.Expression.Literal("Ben"), ast -> ast.setType(Environment.Type.STRING)),
                                init(new Ast.Expression.Literal(BigInteger.TEN), ast -> ast.setType(Environment.Type.INTEGER))
                        ), ast -> ast.setType(Environment.Type.STRING)),
                        "\"Ben\" + 10"
                )
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource
    void testFunctionExpression(String test, Ast.Expression.Function ast, String expected) {
        test(ast, expected);
    }

    private static Stream<Arguments> testFunctionExpression() {
        return Stream.of(
                Arguments.of("Print",
                        // print("Hello, World!")
                        init(new Ast.Expression.Function(Optional.empty(),"print", Arrays.asList(
                                init(new Ast.Expression.Literal("Hello, World!"), ast -> ast.setType(Environment.Type.STRING))
                        )), ast -> ast.setFunction(new Environment.Function("print", "System.out.println", Arrays.asList(Environment.Type.ANY), Environment.Type.NIL, args -> Environment.NIL))),
                        "System.out.println(\"Hello, World!\")"
                )
        );
    }

    /**
     * Helper function for tests, using a StringWriter as the output stream.
     */
    private static void test(Ast ast, String expected) {
        StringWriter writer = new StringWriter();
        new Generator(new PrintWriter(writer)).visit(ast);
        Assertions.assertEquals(expected, writer.toString());
    }

    /**
     * Runs a callback on the given value, used for inline initialization.
     */
    private static <T> T init(T value, Consumer<T> initializer) {
        initializer.accept(value);
        return value;
    }

}
