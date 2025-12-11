package plc.project;

import java.io.PrintWriter;
import java.util.List;
import java.math.BigDecimal;
import java.math.BigInteger;

public final class Generator implements Ast.Visitor<Void> {

    private final PrintWriter writer;
    private int indent = 0;

    public Generator(PrintWriter writer) {
        this.writer = writer;
    }

    public static String getName(String typeName) {
        return Environment.getType(typeName).getJvmName();
    }

    private void print(Object... objects) {
        for (Object object : objects) {
            if (object instanceof Ast) {
                visit((Ast) object);
            } else {
                writer.write(object.toString());
            }
        }
    }

    private void newline(int indent) {
        writer.println();
        for (int i = 0; i < indent; i++) {
            writer.write("    ");
        }
    }

    @Override
    public Void visit(Ast.Source ast) {
        print("public class Main {");
        newline(indent);
        newline(++indent);
        for (Ast.Field field : ast.getFields()) {
            print(field);
            newline(indent);
        }
        print("public static void main(String[] args) {");
        newline(++indent);
        print("System.exit(new Main().main());");
        newline(--indent);
        print("}");

        for (Ast.Method method : ast.getMethods()) {
            newline(--indent);
            newline(++indent);
            print(method);
        }
        newline(--indent);
        newline(indent);
        print("}");
        return null;
    }

    @Override
    public Void visit(Ast.Field ast) {
        print(getName(ast.getTypeName()));
        print(" ", ast.getName());

        if (ast.getValue().isPresent()) {
            print(" = ");
            print(ast.getValue().get());
        }
        print(";");
        return null;
    }

    @Override
    public Void visit(Ast.Method ast) {
        if (ast.getReturnTypeName().isPresent()) {
            print(getName(ast.getReturnTypeName().get()));
        }
        print(" ", ast.getName(), "(");
        int paramSize = ast.getParameters().size();
        int typenameSize = ast.getParameterTypeNames().size();
        if (paramSize == typenameSize && paramSize > 0) {
            for (int i = 0; i < ast.getParameters().size() - 1; i++) {
                print(getName(ast.getParameterTypeNames().get(i)), " ", ast.getParameters().get(i), ", ");
            }
            print(getName(ast.getParameterTypeNames().get(typenameSize - 1)), " ", ast.getParameters().get(paramSize - 1));
        }
        print(") {");
        if (!ast.getStatements().isEmpty()) {
            indent++;
            for (Ast.Statement stmt : ast.getStatements()) {
                newline(indent);
                print(stmt);
            }
            newline(--indent);
        }
        print("}");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Expression ast) {
        print(ast.getExpression());
        print(";");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Declaration ast) {
        if (ast.getTypeName().isPresent()) {
            print(getName(ast.getTypeName().get()));
        } else {
            if (ast.getValue().isPresent()) {
                print(ast.getValue().get().getType().getJvmName());
            } else {
                throw new RuntimeException("Expected Value or Type Declaration.");
            }
        }

        print(" ", ast.getName());
        if (ast.getValue().isPresent()) {
            print(" = ");
            print(ast.getValue().get());
        }
        print(";");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Assignment ast) {
        print(ast.getReceiver());
        print(" = ");
        print(ast.getValue());
        print(";");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.If ast) {
        print("if (");
        print(ast.getCondition());
        print(") {");
        indent++;

        for (Ast.Statement statement : ast.getThenStatements()) {
            newline(indent);
            print(statement);
        }

        if (!ast.getElseStatements().isEmpty()) {
            newline(--indent);
            print("} else {");
            indent++;
            for (Ast.Statement statement : ast.getElseStatements()) {
                newline(indent);
                print(statement);
            }
        }

        newline(--indent);
        print("}");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.For ast) {
        print("for ( ");
        if (ast.getInitialization() != null) {
            print(ast.getInitialization());
        } else {
            print(";");
        }
        print(" ");
        print(ast.getCondition(), ";");
        if (ast.getIncrement() != null) {
            // Increment w/ no ;
            print(' ', ((Ast.Statement.Assignment)ast.getIncrement()).getReceiver());
            print(" = ");
            print(((Ast.Statement.Assignment)ast.getIncrement()).getValue());
        }

        print(" ) {");
        indent++;
        if(!ast.getStatements().isEmpty()) {
            for (Ast.Statement stmt : ast.getStatements()) {
                newline(indent);
                print(stmt);
            }
            newline(--indent);
        }

        print("}");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.While ast) {
        print("while (");
        print(ast.getCondition());
        print(") {");
        indent++;

        if(!ast.getStatements().isEmpty()) {
            for (Ast.Statement stmt : ast.getStatements()) {
                newline(indent);
                print(stmt);
            }
            newline(--indent);
        }
        print("}");
        return null;
    }

    @Override
    public Void visit(Ast.Statement.Return ast) {
        print("return ");
        print(ast.getValue());
        print(";");
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Literal ast) {
        if (ast.getType() == Environment.Type.BOOLEAN) {
            // Bool
            Boolean literal = (Boolean) ast.getLiteral();
            print(literal);
        } else if (ast.getType() == Environment.Type.INTEGER) {
            // Int
            BigInteger literal = (BigInteger) ast.getLiteral();
            print(literal);
        } else if (ast.getType() == Environment.Type.STRING) {
            // String
            String literal = (String) ast.getLiteral();
            print("\"", literal, "\"");
        } else if (ast.getType() == Environment.Type.CHARACTER) {
            // Char
            char literal = (char) ast.getLiteral();
            print("'", literal, "'");
        } else if (ast.getType() == Environment.Type.DECIMAL) {
            // Float
            BigDecimal literal = (BigDecimal) ast.getLiteral();
            print(literal.toString());
        } else {
            throw new RuntimeException("Type Error");
        }
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Group ast) {
        writer.write("(");
        print(ast.getExpression());
        writer.write(")");
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Binary ast) {
        visit(ast.getLeft());
        switch (ast.getOperator()) {
            case "AND":
            case "&&":
                print(" && ");
                break;
            case "OR":
            case "||":
                print(" || ");
                break;
            case "+":
                print(" + ");
                break;
            case "-":
                print(" - ");
                break;
            case "*":
                print(" * ");
                break;
            case "/":
                print(" / ");
                break;
            default:
                print(" ", ast.getOperator(), " ");
                break;
        }
        visit(ast.getRight());
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Access ast) {
        if (ast.getReceiver().isPresent()) {
            print(ast.getReceiver().get());
            print(".");
        }

        print(ast.getName());
        return null;
    }

    @Override
    public Void visit(Ast.Expression.Function ast) {
        Environment.Function func = ast.getFunction();
        if (ast.getReceiver().isPresent()) {
            print(ast.getReceiver().get());
            print(".");
        }

        print(func.getJvmName(), "(");
        List<Ast.Expression> args = ast.getArguments();
        if (!args.isEmpty()) {
            for (int i = 0; i < args.size() - 1; i++) {
                visit(args.get(i));
                print(", ");
            }
            print(args.getLast());
        }
        print(")");
        return null;
    }
}
