#include <iostream>


typedef struct Shape {
    protected:
    double length;
    double width;

    public:
    Shape(double len = 1, double wid = 1) {
        length = len;
        width = wid;
    }

    double get_area() {
        return length * width;
    }

    private:
    int id;

} Shape;


typedef struct Circle: Shape {
    Circle(double wid) {
        width = wid;
    }

    double get_area() {
        return 3.14159 * (width / 2.0) * (width / 2.0);
    }

} Circle;


class Customer{
private:
    friend class GetCustomerData;
    std::string name; 
public:
    Customer(std::string name){
        this->name = name;
    }
};
 
class GetCustomerData{
public:
    static std::string GetName(Customer& customer){
        return customer.name;
    }
};


int main() {

    Shape square(10, 10);
    std::cout << "Square area: " << square.get_area() << std::endl;

    Circle circ(10);
    std::cout << "Circle area: " << circ.get_area() << std::endl;

    Shape rect(10, 15);
    std::cout << "Rectangle area: " << rect.get_area() << std::endl;

    std::cout << "**************************************" << std::endl;

    Customer tom("Tom");
    GetCustomerData getName;
    std::cout << "Name : " << getName.GetName(tom) << "\n";





}
