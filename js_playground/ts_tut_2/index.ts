let sales: number = 123_456_789;
let course: string = "Typescript";
let isPublished: boolean = true;

function render(document: any) {
    console.log(document);
}

// arrays
let numbers: number[] = [1, 2, 3];
numbers.forEach(n => n.toExponential());

// tuples
let user: [number, string] = [1, 'Bob'];

// enums, numbers start at 0 by default
enum Size { Small = 1, Medium, Large };
console.log(Size.Small);
console.log(Size.Large);

// functions
function calculateTax(income: number, taxYear: number = 2022): number {
    if (taxYear < 2022) {
        return income * 0.2;
    }

    else {
        return income * 0.25
    }
}
console.log(calculateTax(10_000, 2023))
console.log(calculateTax(10_000, 2021))

// objects
let employee: {
    readonly id: number,
    name: string,
    retire: (date: Date) => void
} = { id: 1, name: '', retire: (date: Date) => {console.log(date)} };
