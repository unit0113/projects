#include <iostream>
#include <stdexcept>

class Record {
    public:
        // Constructor
        Record() {
        }

        Record(std::string first, std::string last) {
            name_first = first;
            name_last = last;
        }

        // Destructor
        ~Record() {
        }

        std::string get_name() {
            return name_last + ", " + name_first;
        }

        void set_name(std::string last, std::string first) {
            name_first = first;
            name_last = last;
        }

        void set_name(std::string full_name) {
            int delimit_index = full_name.find(", ");
            if (delimit_index == -1) {
                delimit_index = full_name.find(" ");
                if (delimit_index == -1) {
                    throw std::invalid_argument("Invalid Name");
                } else {
                    // First Last
                    name_first = full_name.substr(0, delimit_index);
                    name_last = full_name.substr(delimit_index + 1, full_name.size() - 1);
                }
                
            } else {
                // Last, First
                name_last = full_name.substr(0, delimit_index);
                name_first = full_name.substr(delimit_index + 2, full_name.size() - 1);
            }
        }

    private:
        std::string name_first;
        std::string name_last;
        size_t dob_day;
        size_t dob_month;
        size_t dob_year;
        char sex;
        size_t height;
        size_t weight;


};


int main() {

    Record Jane("Jane", "Goodall");
    std::cout << Jane.get_name() << std::endl;
    Jane.set_name("Molly Doe");
    std::cout << Jane.get_name() << std::endl;
    Jane.set_name("Shelley, Mary");
    std::cout << Jane.get_name() << std::endl;





    return 0;
}