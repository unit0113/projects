#include <iostream>
#include <string>
#include <vector>


class Product {
    private:
        std::string name;
        std::string description;
        size_t quantity;
    
    public:
        Product() {
            std::cout << "Enter name of the product: ";
            std::getline(std::cin, name);
            std::cout << "Enter desciption of the product: ";
            std::getline(std::cin, description);
            std::cout << "Enter the quantity in stock: ";
            std::string q_in;
            std::cin >> q_in;
            std::cin.sync();
            quantity = stoi(q_in);
        }

        Product(std::string name_in, std::string des_in, size_t q_in) {
            name = name_in;
            description = des_in;
            quantity = q_in;
        }

        void print() {
            std::cout << "Name: " << name << ". Num in stock: " << quantity << ". Description: " << description << std::endl;
        }
};


class Inventory {
    private:
        std::vector<Product> products;

    public:
        void add_product() {
            products.push_back(Product());
        }

        void add_product(std::string name_in, std::string des_in, size_t q_in) {
            products.push_back(Product(name_in, des_in, q_in));
        }

        void print() {
            for (Product product: products) {
                product.print();
            }
        }
};


int main() {
    
    Inventory inv = Inventory();
    inv.add_product();
    inv.add_product();
    inv.add_product("Hammer", "Hits stuff", 5292);
    inv.print();

}