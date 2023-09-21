function sayHello(name) {
    console.log('hello ' + name)
}

sayHello('Bob')


// JS Global functions
setTimeout(func, delay, funcParam1, funcParam2) //run function after specific amount of time, returns timeout ID
clearTimeout(timeoutID) // cancels previously established timeout

setInterval(func, delay, funcParam1, funcParam2) //Repeatedly calls a function, return intervalID
clearInterval(intervalID) // canceles a setInterval

// In regular JS, above functions are in the window namespace
// In node, they are in global
