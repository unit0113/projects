// Node path library
const path = require('path');
var pathObj = path.parse(__filename);
console.log(pathObj);

// Node os library
const os = require('os');
var totalMemory = os.totalmem();
var freeMemory = os.freemem();
console.log('Total Memory: ' + totalMemory);
console.log(`Free Memory: ${freeMemory}`);

// Node file system library. Use async methods over sync methods
const fs = require('fs');

var ls = fs.readdirSync('./');
console.log(ls)

fs.readdir('./', function(error, files) {
    if (error) console.log('Error', err);
    else console.log(`Result: [${files}]`)
})
