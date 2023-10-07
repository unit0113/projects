let condition = true;
let x;
let time = 10;
if (condition) {
  x = 0;
} else if (condition && time > 5) {
  x = 5;
} else {
  x = 10;
}

let role = "guest";

switch (role) {
  case "guest":
    console.log("Guest");
    break;
  case "admin":
    console.log("Admin");
    break;
  default:
    console.log("Uknown");
}

for (let i = 0; i < 5; ++i) {
  console.log("Hi!");
}

let i = 0;
while (i < 5) {
  console.log("Hi Again!");
  ++i;
}

i = 0;
do {
  console.log("Hello!");
  ++i;
} while (i < 5);

const person = {
  name: "Bob",
  age: 25,
};

for (let key in person) {
  console.log(key, person[key]);
}

const colors = ["red", "green", "blue"];
for (let index in colors) {
  console.log(index, colors[index]);
}

// for-of
for (let color of colors) {
  console.log(color);
}
