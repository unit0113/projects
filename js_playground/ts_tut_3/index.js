"use strict";
var _a, _b;
let employee = { id: 1, name: '', retire: (date) => { console.log(date); } };
function kgToLbs(weight) {
    if (typeof weight === 'string') {
        weight = parseInt(weight);
    }
    return weight * 2.2;
}
console.log(kgToLbs(10));
console.log(kgToLbs('10kg'));
let textBox = {
    drag: () => { },
    resize: () => { }
};
let quantity = 100;
// nullable types
function greet(name) {
    if (name)
        console.log(name.toUpperCase());
    else
        console.log('Hola!');
}
greet(null);
function getCustomer(id) {
    return id === 0 ? null : { birthday: new Date() };
}
let customer = getCustomer(0);
// optional property access operator
console.log((_a = customer === null || customer === void 0 ? void 0 : customer.birthday) === null || _a === void 0 ? void 0 : _a.getFullYear());
let customer2 = getCustomer(1);
console.log((_b = customer2 === null || customer2 === void 0 ? void 0 : customer2.birthday) === null || _b === void 0 ? void 0 : _b.getFullYear());
