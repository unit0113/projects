const Logger = require('./logger');
const logger = new Logger();
logger.addListener('messageLogged', (eventArg) => {
    console.log('Listener called: ', eventArg);
})
logger.log('message');