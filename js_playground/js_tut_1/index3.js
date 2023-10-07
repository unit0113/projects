const circle = {
  radius: 1,
  location: {
    x: 1,
    y: 1,
  },
  isVisible: true,
  draw: function () {
    console.log("draw");
  },
};

circle.draw();

// Factory
function createCircle(radius, location) {
  return {
    radius,
    location,
    isVisible: true,
    draw() {
      console.log("draw");
    },
  };
}

// Constuctor
function Circle(radius) {
  this.radius = radius;
  this.draw() = function() {
    console.log('draw');
  }
}

const circ = new Circle(3);

for (let key in circ) {
    console.log(key, circ[key]);
}

const circCopy = Object.assign({}, circ);

