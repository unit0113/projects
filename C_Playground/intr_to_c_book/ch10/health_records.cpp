#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <string>
#include <ctime>

class Record {
    // unique account ID

    public:
        // Constructor
        Record() {
            initilizer();
            std::cout << "Unnamed Record Created" << std::endl;
        }

        Record(std::string first, std::string last) {
            initilizer();
            set_name(last, first);
            std::cout << "Record Created for " << name_last << ", " << name_first << std::endl;
        }

        Record(std::string first, std::string last, char sex_input) {
            initilizer();
            set_name(last, first);
            set_sex(sex_input);
            std::cout << "Record Created for " << name_last << ", " << name_first << std::endl;
        }

        // Destructor
        ~Record() {
            if (name_last == "NONE") {
                std::cout << "Unnamed Record Deleted" << std::endl;
            } else {
                std::cout << "Record for " << name_last << ", " << name_first << " Deleted" << std::endl;
            }
        }

        // Initilizer for members
        void initilizer() {
            name_first = "NONE";
            name_last = "NONE";
            dob_day = 0;
            dob_month = 0;
            dob_year = 0;
            sex = 'a';
            height = 0;
            weight = 0;
        }        

        // Name functions
        std::string get_name() {
            if (name_first == "NONE") {
                return "Unnamed Record";
            } else {
                return name_last + ", " + name_first;
            }
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

        // DoB Functions
        std::string get_dob() {
            return months[dob_month] + ' ' + std::to_string(dob_day) + ", " + std::to_string(dob_year);
        }

        size_t get_age() {
            time_t now = time(0);
            tm *now_tm = localtime(&now);
            if (1 + now_tm->tm_mon > dob_month) {
                return 1900 + now_tm->tm_year - dob_year;
            } else {
                return 1900 + now_tm->tm_year - dob_year - 1;
            }
        }
        void print_age() {
            if (dob_year == 0) {
                throw std::invalid_argument("Invalid DoB");
            } else {
                std::cout << get_name() << " is " << get_age() << " years old" << std::endl;
            }
        }

        void print_dob() {
            std::cout << "DoB for " << get_name() << ": " << get_dob() << std::endl;
        }

        void set_dob(size_t day, size_t month, size_t year) {
            dob_day = day;
            dob_month = month;
            dob_year = year;
        }

        // Sex Functions
        char get_sex() {
            if (sex == 'a') {
                throw std::invalid_argument("Record Sex not set");
            } else {
                return sex;
            }
        }

        void set_sex(char sex_input) {
            sex_input = toupper(sex_input);
            if (sex_input == 'F' or sex_input == 'M' ) {
                sex = sex_input;
            } else {
                std::cout << "Invalid Input" << std::endl;
            }
        }

        // Height/Weight Functions
        size_t get_height_inch() {
            if (height == 0) {
                throw std::invalid_argument("Record Height not set");
            } else {
                return height;
            }
        }
        
        std::string get_height_foot_inch() {
            return std::to_string(height / 12) + " feet, " + std::to_string(height % 12) + " inches";
        }

        size_t get_weight() {
            if (weight == 0) {
                throw std::invalid_argument("Record Weight not set");
            } else {
                return weight;
            }
        }

        double get_bmi() {
            if (weight == 0) {
                throw std::invalid_argument("Record Weight not set");
            } else if (height == 0) {
                throw std::invalid_argument("Record Height not set");
            } else {
                return 703.0 * weight / (height * height);
            }
        }

        void print_bmi() {
            std::cout << "BMI for " << get_name() << ": " <<  std::setprecision(4) << get_bmi() << std::endl;;
        }

        void set_height(size_t inches) {
            height = inches;
        }

        void set_weight(size_t lbs) {
            weight = lbs;
        }

    private:
        static size_t account_id;   // Not implemented
        const std::vector<std::string> months = {"NONE", "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

        std::string name_first;
        std::string name_last;
        size_t dob_day;
        size_t dob_month;
        size_t dob_year;
        char sex;
        size_t height;
        size_t weight;

        // Disable Copy Constructor
        Record(const Record&);
};


int main() {

    Record Jane("Jane", "Goodall");
    std::cout << Jane.get_name() << std::endl;
    Jane.set_name("Molly Doe");
    std::cout << Jane.get_name() << std::endl;
    Jane.set_name("Shelley, Mary");
    std::cout << Jane.get_name() << std::endl;
    Jane.set_dob(3, 4, 1934);
    std::cout << Jane.get_dob() << std::endl;
    Jane.print_dob();
    Jane.print_age();

    Record Mike;
    std::cout << Mike.get_name() << std::endl;
    Mike.set_name("Mike Wyzgowski");
    Mike.set_height(72);
    Mike.set_weight(180);
    Mike.print_bmi();

    Record Mary("Mary", "Sue", 'f');
    std::cout << Mary.get_sex() << std::endl;

    return 0;
}