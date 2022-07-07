import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology, login_required, lookup, usd

# Configure application
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

# Make sure API key is set
if not os.environ.get("API_KEY"):
    raise RuntimeError("API_KEY not set")


@app.route("/", methods=["GET", "POST"])
@login_required
def index():

    if request.method == "GET":
        # Pull transaction history
        rows = db.execute("""
            SELECT symbol, SUM(shares) as totalShares, price, timestamp
            FROM transactions
            WHERE user_id = :user_id
            GROUP BY symbol
            HAVING SUM(shares) > 0;
        """, user_id=session["user_id"])

        # Get history into a useable form
        total = 0
        holdings = []
        for row in rows:
            stock = lookup(row["symbol"])
            holdings.append({
                "symbol": row["symbol"],
                "name": stock["name"],
                "shares": row["totalShares"],
                "gain": round((((stock["price"] * row["totalShares"]) - float(row["price"])) / float(row["price"])) * 100, 2),
                "costBasis": usd(row["price"]),
                "curValue": usd(stock["price"] * row["totalShares"])
            })

            # Update total
            total += stock["price"] * row["totalShares"]

        # Pull cash value
        cashPull = db.execute("SELECT cash FROM users WHERE id=:id", id=session["user_id"])
        cash = cashPull[0]["cash"]
        total += cash

        return render_template("index.html", holdings=holdings, cash=usd(cash), total=usd(total))

    else:
        # Get deposit and current cash balance
        deposit = request.form.get("cashDeposit")
        cashPull = db.execute("SELECT cash FROM users WHERE id=:id", id=session["user_id"])
        cash = cashPull[0]["cash"]
        updatedCash = int(cash) + int(deposit)

        # Update cash balance
        db.execute("UPDATE users SET cash=:updatedCash WHERE id=:id",
                   updatedCash=updatedCash,
                   id=session["user_id"]
                   )

        flash("Deposit Successful!")
        return redirect("/")


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":

        # Ensure ticker was submitted
        if not request.form.get("symbol"):
            return apology("Must provide ticker", 400)

        # Ensure number of shares was submitted
        if not request.form.get("shares"):
            return apology("Must provide number of shares", 400)

        # Ensure number of shares is int
        if not request.form.get("shares").isdigit():
            return apology("Share count must be an integer", 400)

        # Get data
        symbol = request.form.get("symbol").upper()
        shares = int(request.form.get("shares"))
        stock = lookup(symbol)

        # Check if real ticker
        if stock is None:
            return apology("Invalid Ticker", 400)

        costBasis = stock['price']
        price = shares * costBasis

        # Get cash balance
        cashPull = db.execute("SELECT cash FROM users WHERE id=:id", id=session["user_id"])
        cash = cashPull[0]["cash"]
        updatedCash = cash - price

        # Check if enough cash
        if updatedCash < 0:
            return apology("Not enough cash for purchase", 400)

        # Update cash balance
        db.execute("UPDATE users SET cash=:updatedCash WHERE id=:id",
                   updatedCash=updatedCash,
                   id=session["user_id"]
                   )

        # Update transactions database
        db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES(?, ?, ?, ?)",
                   session["user_id"], symbol, shares, price)
        flash("Purchase Successful!")

        return redirect("/")

    else:
        return render_template("buy.html")


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    rows = db.execute("""
        SELECT symbol, shares, price, timestamp
        FROM transactions
        WHERE user_id=:user_id
        ORDER BY id
    """, user_id=session["user_id"])

    transactions = []
    for row in rows:
        transactions.append({
            "symbol": row["symbol"],
            "shares": row["shares"],
            "price": usd(row["price"]),
            "timestamp": row["timestamp"]
        })

    return render_template("history.html", transactions=transactions)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Query database for username
        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", 400)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":

        # Ensure ticker was submitted
        if not request.form.get("symbol"):
            return apology("must provide ticker", 400)

        #  Get data
        symbol = request.form.get("symbol").upper()
        stock = lookup(symbol)

        # Check if real ticker
        if stock is None:
            return apology("Invalid Ticker", 400)

        return render_template("quoted.html", stockData={
            'name': stock['name'],
            'price': usd(stock['price']),
            'symbol': stock['symbol']
        })

    else:
        return render_template("quote.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 400)

        # Ensure password was submitted twice
        elif not request.form.get("confirmation"):
            return apology("must confirm password", 400)

        # Get info from forms
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        # Check passwords
        if password != confirmation:
            return apology("passwords must match", 400)

        # Check if user already exists
        prevuser = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))
        if len(prevuser) == 1:
            return apology("username unavailable, please select another", 400)

        # Add user to database
        hashpassword = generate_password_hash(request.form.get("password"))
        key = db.execute("INSERT INTO users (username, hash) VALUES(?, ?)", username, hashpassword)

        # Remember which user has logged in
        session["user_id"] = key

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""
    if request.method == "POST":

        # Ensure ticker was submitted
        if not request.form.get("symbol"):
            return apology("must provide ticker", 403)

        # Ensure number of shares was submitted
        if not request.form.get("shares"):
            return apology("must provide number of shares", 403)

        # Get data
        symbol = request.form.get("symbol").upper()
        shares = int(request.form.get("shares"))
        stock = lookup(symbol)

        # Check if real ticker
        if stock is None:
            return apology("Invalid Ticker", 403)

        costBasis = stock['price']
        price = shares * costBasis

        # Check if enough shares and if shares owned
        rows = db.execute(
            "SELECT symbol, SUM(shares) as sharesTotal FROM transactions WHERE user_id=:id GROUP BY symbol HAVING sharesTotal > 0", id=session["user_id"])
        holdFlag = False
        for row in rows:
            if row["symbol"] == symbol:
                holdFlag = True
                if shares > row["sharesTotal"]:
                    return apology("Not enough shares", 400)
                break
        if holdFlag == False:
            return apology("No shares held", 400)

        # Get cash balance
        cashPull = db.execute("SELECT cash FROM users WHERE id=:id", id=session["user_id"])
        cash = cashPull[0]["cash"]
        updatedCash = cash + price

        # Update cash balance
        db.execute("UPDATE users SET cash=:updatedCash WHERE id=:id",
                   updatedCash=updatedCash,
                   id=session["user_id"]
                   )

        # Update transactions database
        shares = -1 * shares
        db.execute("INSERT INTO transactions (user_id, symbol, shares, price) VALUES(?, ?, ?, ?)",
                   session["user_id"], symbol, shares, price)
        flash("Sale Successful!")

        return redirect("/")

    else:
        rows = db.execute(
            "SELECT symbol FROM transactions WHERE user_id=:user_id GROUP BY symbol HAVING SUM(shares) > 0", user_id=session["user_id"])
        return render_template("sell.html", holdings=[row["symbol"] for row in rows])


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return apology(e.name, e.code)


# Listen for errors
for code in default_exceptions:
    app.errorhandler(code)(errorhandler)
