// Function declaration (can be called before being definined)
function walk() {
  console.log("walking");
}

// function expression (lambda)
let run = function () {
  console.log("running");
};

function sum(a, b) {
  return a + b;
}

function sumAll() {
  sum = 0;
  for (let num of arguments) {
    sum += num;
  }
  return sum;
}

console.log(sumAll(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));

// rest operator
function sumAll2(...args) {
  args.reduce((a, b) => a + b);
}

function interest(principal, rate = 3.5, years = 5) {
  return ((principal * rate) / 100) * years;
}

const person = {
  firstName: "Billy-Bob",
  lastName: "Joe",
  get fullName() {
    return "${person.firstName}, ${person.lastName}";
  },
  set fullName(name) {
    if (typeof name !== "string") {
      throw new Error("Value must be a string");
    }
    const parts = name.split(" ");

    if (parts.length !== 2) return;
    this.firstName = parts[0];
    this.lastName = parts[1];
  },
};
console.log(person.fullName());
person.fullName = "John Smith";
console.log(person.fullName());
try {
  person.fullName = null;
} catch (e) {
  alert(e);
}
