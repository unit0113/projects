type Employee = {
    readonly id: number,
    name: string,
    retire: (date: Date) => void
}

let employee: Employee = { id: 1, name: '', retire: (date: Date) => {console.log(date)} };

function kgToLbs(weight: number | string): number {
    if (typeof weight === 'string') {
        weight = parseInt(weight);
    }
    return weight * 2.2;
}
console.log(kgToLbs(10));
console.log(kgToLbs('10kg'));

type Draggable = {
    drag: () => void
};

type Resizeable = {
    resize: () => void
};

// type unions/type intersections
type UIWidget = Draggable & Resizeable
let textBox: UIWidget = {
    drag: () => {},
    resize: () => {}
};

// literal types
type Quantity = 50 | 100;
let quantity: Quantity = 100;

type Metric = 'cm' | 'inch';

// nullable types
function greet(name: string | null | undefined): void {
    if (name)
        console.log(name.toUpperCase());
    else
        console.log('Hola!');
}
greet(null);

// optional chaining
type Customer = {
    birthday: Date
};

function getCustomer(id: number): Customer | null | undefined {
    return id === 0 ? null : { birthday: new Date() };
}

let customer = getCustomer(0);
// optional property access operator
console.log(customer?.birthday?.getFullYear());
let customer2 = getCustomer(1);
console.log(customer2?.birthday?.getFullYear());

// optional element access operator
// customer?.[0]

// optional call
let log: any = null;
log?.('a');