const logger = require('./logger')  // Node assumes .js extension, storing as const vs var is best practice, assures this files does not change imported module

logger.log('Hello World!')