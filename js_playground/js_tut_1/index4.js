const numbers = [3, 4];

numbers.push(5, 6);

numbers.unshift(7);

numbers.splice(2, 0, "a", "b");

console.log(numbers);

console.log(numbers.indexOf(6));

numbers = [1, 2, 3, 1, 4];

console.log(numbers.lastIndexOf(1));
console.log(numbers.includes(1));

const courses = [
  { id: 1, name: "a" },
  { id: 2, name: "b" },
];

console.log(courses.includes({ id: 1, name: "a" })); //false
console.log(courses.find({ id: 1, name: "a" })); // true

const course = courses.find(function (course) {
  return course.name === "a";
});
console.log(course);

numbers = [1, 2, 3, 4];

const last = numbers.pop();
const first = numbers.shift();
numbers.splice(2, 2);
console.log(numbers);

const front = [1, 2, 3];
const back = [4, 5, 6];

const combined = front.concat(back);

const slice = combined.slice(2);

// Unpack
combined = [...front, "a", ...back, "b"];
const copy = [...combined];

numbers.forEach((number, index) => console.log(index, number));

for (let number of numbers) {
  console.log(number);
}

const joined = numbers.join(",");
const message = "This is my first message";
const parts = message.split(" ");

numbers = [3, 6, 1, 7, 4];
numbers.sort();
numbers.reverse();

courses.sort(function (a, b) {
  if (a.name < b.name) return -1;
  if (a.name > b.name) return 1;
  return 0;
});

numbers = [1, -1, 3, 5];
const allPositive = numbers.every(function (value) {
  return value >= 0;
}); // false

const somePositive = numbers.some(function (value) {
  return value >= 0;
}); // true

const positive = numbers.filter((num) => num >= 0);

const items = positive.map((n) => "<li>" + n + "</li>");
const html = "<ul>" + items.join("") + "</ul>";

const sum = numbers.reduce(
  (accumulator, currentValue) => accumulator + currentValue
);
