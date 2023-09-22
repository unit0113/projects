"use strict";
let sales = 123456789;
let course = "Typescript";
let isPublished = true;
function render(document) {
    console.log(document);
}
// arrays
let numbers = [1, 2, 3];
numbers.forEach(n => n.toExponential());
// tuples
let user = [1, 'Bob'];
// enums, numbers start at 0 by default
var Size;
(function (Size) {
    Size[Size["Small"] = 1] = "Small";
    Size[Size["Medium"] = 2] = "Medium";
    Size[Size["Large"] = 3] = "Large";
})(Size || (Size = {}));
;
console.log(Size.Small);
console.log(Size.Large);
// functions
function calculateTax(income, taxYear = 2022) {
    if (taxYear < 2022) {
        return income * 0.2;
    }
    else {
        return income * 0.25;
    }
}
console.log(calculateTax(10000, 2023));
console.log(calculateTax(10000, 2021));
