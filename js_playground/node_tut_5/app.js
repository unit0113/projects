const http = require('http');
const server = http.createServer((request, response) => {
    if (request.url === '/') {
        response.write('Hello World');
        response.end();
    }

    if (request.url === '/api/courses') {
        response.write(JSON.stringify([1,2,3]));
        response.end();
    }
});

/*server.addListener('connection', (socket) => {
    console.log('New connection');
})*/



server.listen(3000);    // listen to port
console.log('Listening on port 3000...');