// Module for logging messages

var url = 'http://mylogger.io/log' // Made-up website of logger
function log(message) {
    console.log(message)
}

module.exports.log = log
module.exports.endPoint = url   // Normally keep private

// exports.XXXX is the name used to access the object

// module.exports = log // exports a single function, whole module is single function