#include <iostream>

class Animal{
    private:
        std::string name;
        double height;
        double weight;

        static int numOfAnimals;
 

    public:
        std::string GetName(){return name;}
        
        void SetName(std::string name){this->name = name;}
        double GetHeight(){return height;}
        void SetHeight(double height){this->height = height;}
        double GetWeight(){return weight;}
        void SetWeight(double weight){this->weight = weight;}
        
        // You can declare function prototypes
        void SetAll(std::string, double, double);
        
        // Constructor prototype
        Animal(std::string, double, double);
        
        // Create an overloaded constructor for when no data is passed
        Animal();
        ~Animal();
        
        // Static methods can only access static fields
        static int GetNumOfAnimals(){return numOfAnimals;}
        
        // Created to be overwritten
        void ToString();
    
};


// Refer to class fields and methods with ::
int Animal::numOfAnimals = 0;
 
// Define the prototype method
void Animal::SetAll(std::string name, double height, double weight){
    this->name = name;
    this->weight = weight;
    this->height = height;
    Animal::numOfAnimals++;
}
 
// Define the constructor
Animal::Animal(std::string name, double height, double weight){
    this->name = name;
    this->weight = weight;
    this->height = height;
    Animal::numOfAnimals++;
}
 
Animal::Animal(){
    this->name = "";
    this->weight = 0;
    this->height = 0;
    Animal::numOfAnimals++;
}
 
Animal::~Animal(){
    std::cout << "Animal " << this -> name << 
            " destroyed\n";
}
 
void Animal::ToString(){
    std::cout << this -> name << " is " << 
            this -> height <<
            " cms tall and " << this -> weight <<
            " kgs in weight\n";
}
 
// Through inheritance a class inherits all fields and methods
// defined by the super, or inherited from class
class Dog: public Animal{
private:
    std::string sound = "Wooof";
public:
    // You can access to the private field name
    // by calling GetName()
    void MakeSound(){ 
        std::cout << "The dog " << this->GetName() << " says " << this->sound << "\n";
    }
    
    // The Dogs constructor
    Dog(std::string, double, double, std::string);
    
    // The default constructor calls Animals default constructor
    Dog(): Animal(){};
    
    // Overwrite ToString
    void ToString();
    
};
 
// Calls the superclasses constructor to handle initalization
Dog::Dog(std::string name, double height, 
double weight, std::string sound) :
Animal(name, height, weight){
    this -> sound = sound;
}
 
// Overwrite ToString
void Dog::ToString(){
    // Because the attributes were private in Animal they must be retrieved 
    // by called the get methods
    std::cout << this -> GetName() << " is " << this -> GetHeight() << 
            " cms tall, " << this -> GetWeight() << 
            " kgs in weight and says " << this -> sound << "\n";
}


class Warrior {
    public:

        Warrior(std::string name_in, int health_in, int pow_in, int blk_in) {
            name = name_in;
            health = health_in;
            power = pow_in;
            blk = blk_in;
        }

        int attack() {
            return std::rand() % power;
        }

        int block() {
            return std::rand() % blk;
        }

        bool take_hit(int dmg) {
            health -= dmg;
            return health < 0;
        }

        std::string get_name() {
            return name;
        }

    private:
        std::string name;
        int health;
        int power;
        int blk;

};


class Battle {
    public:
        static void StartFight(Warrior &w1, Warrior &w2) {
            while (true) {
                if (Battle::attack_round(w1, w2)) {
                    std::cout << "Game over!" << std::endl;
                    break;
                } else if (Battle::attack_round(w2, w1)) {
                    std::cout << "Game over!" << std::endl;
                    break;
                }
            }
        }

        static bool attack_round(Warrior &w1, Warrior &w2) {
            int dmg = std::max(w1.attack() - w2.block(), 0);
            std::cout << w1.get_name() << " hits " << w2.get_name() << " for " << dmg << " damage" << std::endl;
            return w2.take_hit(dmg);
        }
};











int main() {

    // Create object without setting values in constructor
    Animal fred;
    fred.SetHeight(33);
    fred.SetWeight(10);
    fred.SetName("Fred");
    // Get the values for the Animal
    fred.ToString();
    
    fred.SetAll("Fred", 34, 12);
    
    fred.ToString();
    
    // Setting values with constructor
    Animal tom("Tom", 36, 15);
    tom.ToString();
    
    // Demonstrate inherited Dog class
    Dog spot("Spot", 38, 16, "Wooof");
    
    // See different output from overwritten ToString()
    spot.ToString();
    
    // Call static methods by using the class name to
    // show the total Animals created
    std::cout << "Number of Animals " << 
            Animal::GetNumOfAnimals() << "\n";


    std::cout << "***********************************************" << std::endl;
    srand(time(NULL));
    Warrior thor("Thor", 100, 30, 15);
    Warrior hulk("Hulk", 135, 25, 10);

    Battle::StartFight(thor, hulk);


}
