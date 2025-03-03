package plc.project;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * See the Parser assignment specification for additional notes on each AST class.
 */
public abstract class Ast {

    public static final class Source extends Ast {

        private final List<Field> fields;
        private final List<Method> methods;

        public Source(List<Field> fields, List<Method> methods) {
            this.fields = fields;
            this.methods = methods;
        }

        public List<Field> getFields() {
            return fields;
        }

        public List<Method> getMethods() {
            return methods;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof Source &&
                    fields.equals(((Source) obj).fields) &&
                    methods.equals(((Source) obj).methods);
        }

        @Override
        public String toString() {
            return "Ast.Source{" +
                    "fields=" + fields +
                    "methods=" + methods +
                    '}';
        }

    }

    public static final class Field extends Ast {

        private final String name;
        private final boolean constant;
        private final Optional<Ast.Expression> value;

        public Field(String name, boolean constant, Optional<Ast.Expression> value) {
            this.name = name;
            this.constant = constant;
            this.value = value;
        }

        public String getName() {
            return name;
        }

        public boolean getConstant() {
            return constant;
        }

        public Optional<Ast.Expression> getValue() {
            return value;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof Field &&
                    name.equals(((Field) obj).name) &&
                    (constant == ((Field) obj).constant) &&
                    value.equals(((Field) obj).value);
        }

        @Override
        public String toString() {
            return "Ast.Field{" +
                    "name='" + name + '\'' +
                    ", constant='" + constant + '\'' +
                    ", value=" + value +
                    '}';
        }

    }

    public static final class Method extends Ast {

        private final String name;
        private final List<String> parameters;
        private final List<Statement> statements;

        public Method(String name, List<String> parameters, List<Statement> statements) {
            this.name = name;
            this.parameters = parameters;
            this.statements = statements;
        }

        public String getName() {
            return name;
        }

        public List<String> getParameters() {
            return parameters;
        }

        public List<Statement> getStatements() {
            return statements;
        }

        @Override
        public boolean equals(Object obj) {
            return obj instanceof Method &&
                    name.equals(((Method) obj).name) &&
                    parameters.equals(((Method) obj).parameters) &&
                    statements.equals(((Method) obj).statements);
        }

        @Override
        public String toString() {
            return "Ast.Method{" +
                    "name='" + name + '\'' +
                    ", parameters=" + parameters +
                    ", statements=" + statements +
                    '}';
        }

    }

    public static abstract class Statement extends Ast {

        public static final class Expression extends Statement {

            private final Ast.Expression expression;

            public Expression(Ast.Expression expression) {
                this.expression = expression;
            }

            public Ast.Expression getExpression() {
                return expression;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Ast.Statement.Expression &&
                        expression.equals(((Ast.Statement.Expression) obj).expression);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Expression{" +
                        "expression=" + expression +
                        '}';
            }

        }

        public static final class Declaration extends Statement {

            private String name;
            private Optional<Ast.Expression> value;

            public Declaration(String name, Optional<Ast.Expression> value) {
                this.name = name;
                this.value = value;
            }

            public String getName() {
                return name;
            }

            public Optional<Ast.Expression> getValue() {
                return value;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Declaration &&
                        name.equals(((Declaration) obj).name) &&
                        value.equals(((Declaration) obj).value);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Declaration{" +
                        "name='" + name + '\'' +
                        ", value=" + value +
                        '}';
            }

        }

        public static final class Assignment extends Statement {

            private final Ast.Expression receiver;
            private final Ast.Expression value;

            public Assignment(Ast.Expression receiver, Ast.Expression value) {
                this.receiver = receiver;
                this.value = value;
            }

            public Ast.Expression getReceiver() {
                return receiver;
            }

            public Ast.Expression getValue() {
                return value;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Assignment &&
                        receiver.equals(((Assignment) obj).receiver) &&
                        value.equals(((Assignment) obj).value);
            }

            @Override
            public final String toString() {
                return "Ast.Statement.Assignment{" +
                        "receiver=" + receiver +
                        ", value=" + value +
                        '}';
            }

        }

        public static final class If extends Statement {

            private final Ast.Expression condition;
            private final List<Statement> thenStatements;
            private final List<Statement> elseStatements;


            public If(Ast.Expression condition, List<Statement> thenStatements, List<Statement> elseStatements) {
                this.condition = condition;
                this.thenStatements = thenStatements;
                this.elseStatements = elseStatements;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public List<Statement> getThenStatements() {
                return thenStatements;
            }

            public List<Statement> getElseStatements() {
                return elseStatements;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof If &&
                        condition.equals(((If) obj).condition) &&
                        thenStatements.equals(((If) obj).thenStatements) &&
                        elseStatements.equals(((If) obj).elseStatements);
            }

            @Override
            public String toString() {
                return "Ast.Statement.If{" +
                        "condition=" + condition +
                        ", thenStatements=" + thenStatements +
                        ", elseStatements=" + elseStatements +
                        '}';
            }

        }

        public static final class For extends Statement {

            private final Statement initialization;
            private final Ast.Expression condition;
            private final Statement increment;
            private final List<Statement> statements;

            public For(Statement initialization, Ast.Expression condition, Statement increment, List<Statement> statements) {
                this.initialization = initialization;
                this.condition = condition;
                this.increment = increment;
                this.statements = statements;
            }

            public Ast.Statement getInitialization() {
                return initialization;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public Ast.Statement getIncrement() {
                return increment;
            }

            public List<Statement> getStatements() {
                return statements;
            }

            @Override
            public boolean equals(Object obj) {

                boolean init, incr;
                For myFor = null;

                if (obj instanceof For) {
                    myFor = (For) obj;
                } else {
                    return false;
                }

                if (initialization == null || myFor.initialization == null) {
                    init = initialization == myFor.initialization;
                } else {
                    init = initialization.equals(myFor.initialization);
                }

                if (increment == null || myFor.increment == null) {
                    incr = increment == myFor.increment;
                } else {
                    incr = increment.equals(myFor.increment);
                }

                return  init &&
                        condition.equals(myFor.condition) &&
                        incr &&
                        statements.equals(myFor.statements);
            }

            @Override
            public String toString() {
                return "For{" +
                        "initialization=" + initialization +
                        ", condition=" + condition +
                        ", increment=" + increment +
                        ", statements=" + statements +
                        '}';
            }

        }

        public static final class While extends Statement {

            private final Ast.Expression condition;
            private final List<Statement> statements;

            public While(Ast.Expression condition, List<Statement> statements) {
                this.condition = condition;
                this.statements = statements;
            }

            public Ast.Expression getCondition() {
                return condition;
            }

            public List<Statement> getStatements() {
                return statements;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof While &&
                        condition.equals(((While) obj).condition) &&
                        statements.equals(((While) obj).statements);
            }

            @Override
            public String toString() {
                return "Ast.Statement.While{" +
                        "condition=" + condition +
                        ", statements=" + statements +
                        '}';
            }

        }

        public static final class Return extends Statement {

            private final Ast.Expression value;

            public Return(Ast.Expression value) {
                this.value = value;
            }

            public Ast.Expression getValue() {
                return value;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Return &&
                        value.equals(((Return) obj).value);
            }

            @Override
            public String toString() {
                return "Ast.Statement.Return{" +
                        "value=" + value +
                        '}';
            }

        }

    }

    public static abstract class Expression extends Ast {

        public static final class Literal extends Ast.Expression {

            private final Object literal;

            public Literal(Object literal) {
                this.literal = literal;
            }

            public Object getLiteral() {
                return literal;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Literal &&
                        Objects.equals(literal, ((Literal) obj).literal);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Literal{" +
                        "literal=" + literal +
                        '}';
            }

        }

        public static final class Group extends Ast.Expression {

            private final Ast.Expression expression;

            public Group(Ast.Expression expression) {
                this.expression = expression;
            }

            public Ast.Expression getExpression() {
                return expression;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Group &&
                        expression.equals(((Group) obj).expression);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Group{" +
                        "expression=" + expression +
                        '}';
            }

        }

        public static final class Binary extends Ast.Expression {

            private final String operator;
            private final Ast.Expression left;
            private final Ast.Expression right;

            public Binary(String operator, Ast.Expression left, Ast.Expression right) {
                this.operator = operator;
                this.left = left;
                this.right = right;
            }

            public String getOperator() {
                return operator;
            }

            public Ast.Expression getLeft() {
                return left;
            }

            public Ast.Expression getRight() {
                return right;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Binary &&
                        operator.equals(((Binary) obj).operator) &&
                        left.equals(((Binary) obj).left) &&
                        right.equals(((Binary) obj).right);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Binary{" +
                        "operator='" + operator + '\'' +
                        ", left=" + left +
                        ", right=" + right +
                        '}';
            }

        }

        public static final class Access extends Ast.Expression {

            private final Optional<Ast.Expression> receiver;
            private final String name;

            public Access(Optional<Ast.Expression> receiver, String name) {
                this.receiver = receiver;
                this.name = name;
            }

            public Optional<Ast.Expression> getReceiver() {
                return receiver;
            }

            public String getName() {
                return name;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Access &&
                        receiver.equals(((Access) obj).receiver) &&
                        name.equals(((Access) obj).name);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Access{" +
                        "receiver=" + receiver +
                        ", name='" + name + '\'' +
                        '}';
            }

        }

        public static final class Function extends Ast.Expression {

            private final Optional<Ast.Expression> receiver;
            private final String name;
            private final List<Ast.Expression> arguments;

            public Function(Optional<Ast.Expression> receiver, String name, List<Ast.Expression> arguments) {
                this.receiver = receiver;
                this.name = name;
                this.arguments = arguments;
            }

            public Optional<Ast.Expression> getReceiver() {
                return receiver;
            }

            public String getName() {
                return name;
            }

            public List<Ast.Expression> getArguments() {
                return arguments;
            }

            @Override
            public boolean equals(Object obj) {
                return obj instanceof Function &&
                        receiver.equals(((Function) obj).receiver) &&
                        name.equals(((Function) obj).name) &&
                        arguments.equals(((Function) obj).arguments);
            }

            @Override
            public String toString() {
                return "Ast.Expression.Function{" +
                        "receiver=" + receiver +
                        ", name='" + name + '\'' +
                        ", arguments=" + arguments +
                        '}';
            }

        }

    }
}
