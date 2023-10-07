let firstName = "Bob";
console.log(firstName);
const interestRate = 0.25;
let selected = null;
let lastName = undefined;

let person = {
  firstName: "Billy",
  LastName: "Bob",
  Age: 3,
};
console.log(person.firstName); //better
console.log(person[age]);

let colors = ["red", "blue", "green"];
colors[3] = "purple";

function greet(name) {
  console.log("Hello there " + name);
}

function square(number) {
  return number * number;
}

let x = 10;
let y = 3;
console.log(x + y);
console.log(x - y);
console.log(x * y);
console.log(x / y);
console.log(x % y);
console.log(x ** y);

x++;
--x;

x += 1;

//strict equality
x = 1;
console.log(x === 1);
// lose equality
console.log(x == "1");
console.log(x !== 1);

let z = x > 0 ? 5 : 10;

console.log(true && false);
console.log(true || false);

let allow = true;
allow = !allow;

console.log(false || "Bob"); // prints Bob
