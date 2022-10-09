#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <initializer_list>
#include <fstream>
#include <sstream>
#include <stdexcept>
using namespace std;

struct LegoSet {
    int m_number;
    string m_theme;
    string m_name;
    int m_minifigs;
    int m_pieces;
    double m_price;

    LegoSet(int number, string theme, string name, int minifigs, int pieces, double price) {
        m_number = number;
        m_theme = theme;
        m_name = name;
        m_minifigs = minifigs;
        m_pieces = pieces;
        m_price = price;
    }
};

vector<LegoSet> loadfile(std::initializer_list<string> args);
void mostExpensive(vector<LegoSet> sets);
void mostPieces(vector<LegoSet> sets);
void printSetInfo(LegoSet set);
void printSetInfoShort(LegoSet set);
void nameMatch(vector<LegoSet> sets, string phrase);
void phraseMatch(vector<LegoSet> sets, string phrase);
void printMatches(vector<LegoSet> sets, string phrase);
void printPartCountInfo(vector<LegoSet> sets);
void printPriceInfo(vector<LegoSet> sets);
void printMinifigInfo(vector<LegoSet> sets);
void printOneOfEverything(vector<LegoSet> sets);


int main() {
    cout << std::fixed << setprecision(2);
    cout << "Which file(s) to open?\n";
    cout << "1. lego1.csv" << endl;
    cout << "2. lego2.csv" << endl;
    cout << "3. lego3.csv" << endl;
    cout << "4. All 3 files" << endl;
    int option;
    cin >> option;
    vector<LegoSet> legoSets;

    /*======= Load data from file(s) =======*/
    switch (option) {
        case 1:
            legoSets = loadfile({"lego1.csv"});
            break;
        case 2:
            legoSets = loadfile({"lego2.csv"});
            break;
        case 3:
            legoSets = loadfile({"lego3.csv"});
            break;
        case 4:
            legoSets = loadfile({"lego1.csv", "lego2.csv", "lego3.csv"});
            break;
        default:
            throw std::runtime_error("Invalid Input");
            break;
    }

    /*======= Print out how many sets were loaded =======*/
    cout << "Total number of sets: " << legoSets.size() << endl;;

    cout << "1. Most Expensive set" << endl;
    cout << "2. Largest piece count" << endl;
    cout << "3. Search for set name containing..." << endl;
    cout << "4. Search themes..." << endl;
    cout << "5. Part count information" << endl;
    cout << "6. Price information" << endl;
    cout << "7. Minifigure information" << endl;
    cout << "8. If you bought one of everything..." << endl;
    cout << "0. Exit" << endl;
   
	int choice;
	cin >> choice;
    cin.get();  // Clear newline character for any later input
   
    /*======= Based on the choice, execute the appropriate task and show the results =======*/
    switch (choice) {
        case 1:
            mostExpensive(legoSets);
            break;
        case 2:
            mostPieces(legoSets);
            break;
        case 3:
            {
                string phrase;
                cout << "Input a search string: ";
                getline(cin, phrase);
                nameMatch(legoSets, phrase);
                break;
            }
            
        case 4:
            {
                string phrase;
                cout << "Input a search string: ";
                getline(cin, phrase);
                phraseMatch(legoSets, phrase);
                break;
            }
        case 5:
            printPartCountInfo(legoSets);
            break;
        case 6:
            printPriceInfo(legoSets);
            break;
        case 7:
            printMinifigInfo(legoSets);
            break;
        case 8:
            printOneOfEverything(legoSets);
            break;
        case 0:
            break;
        default:
            throw std::runtime_error("Invalid Input");
            break;
    }
   
	return 0;
}


vector<LegoSet> loadfile(std::initializer_list<string> args) {
    ifstream inFile;
    vector<LegoSet> setData;
    string inLine;
    istringstream inStream;
    string token, theme, name;
    int number, minifigs, pieces;
    double price;

    // Load each file
    for (const auto& file: args) {
        inFile.open(file);
        if (!inFile) throw std::runtime_error("Error: File not Found");

        // Parse each line, skipping the first line
        getline(inFile, inLine);
        while (getline(inFile, inLine)) {
            inStream.clear();
            inStream.str(inLine);

            // Parse line
            getline(inStream, token, ',');
            number = stoi(token);
            getline(inStream, theme, ',');
            getline(inStream, name, ',');
            getline(inStream, token, ',');
            minifigs = stoi(token);
            getline(inStream, token, ',');
            pieces = stoi(token);
            getline(inStream, token, ',');
            price = stod(token);

            // Add to return vector
            setData.push_back(LegoSet(number, theme, name, minifigs, pieces, price));
        }
        inFile.close();
    }
    return setData;
}


void mostExpensive(vector<LegoSet> sets) {
    int maxIndex{};
    for (size_t i = 1; i < sets.size(); ++i) {
        if (sets[i].m_price > sets[maxIndex].m_price) {
            maxIndex = i;
        }
    }

    cout << "The most expensive set is:\n";
    printSetInfo(sets[maxIndex]);
}


void mostPieces(vector<LegoSet> sets) {
    int maxIndex{};
    for (size_t i = 1; i < sets.size(); ++i) {
        if (sets[i].m_pieces > sets[maxIndex].m_pieces) {
            maxIndex = i;
        }
    }

    cout << "The set with the highest parts count:\n";
    printSetInfo(sets[maxIndex]);
}


void printSetInfo(LegoSet set) {
    cout << "Name: " << set.m_name << endl;
    cout << "Number: " << set.m_number << endl;
    cout << "Theme: " << set.m_theme << endl;
    cout << "Price: $" << set.m_price << endl;
    cout << "Minifigures: " << set.m_minifigs << endl;
    cout << "Piece count: " << set.m_pieces << endl;
}


void printSetInfoShort(LegoSet set) {
    cout << set.m_number << ' ' << set.m_name << " $" << set.m_price << endl;
}


void nameMatch(vector<LegoSet> sets, string phrase) {
    vector<LegoSet> matches;
    for (const auto& set: sets) {
        if (set.m_name.find(phrase) != string::npos) {
            matches.push_back(set);
        }
    }

    printMatches(matches, phrase);
}


void phraseMatch(vector<LegoSet> sets, string phrase) {
    vector<LegoSet> matches;
    for (const auto& set: sets) {
        if (set.m_name.find(phrase) != string::npos) {
            matches.push_back(set);
        }
    }

    printMatches(matches, phrase);
}


void printMatches(vector<LegoSet> sets, string phrase) {
    if (sets.size() == 0) {
        cout << "No sets found matching that search term\n";
    } else {
        cout << "Sets matching \"" << phrase << "\":\n";
        for (const auto& set: sets) {
            printSetInfoShort(set);
        }
    }
}


void printPartCountInfo(vector<LegoSet> sets) {
    int sum{};
    for (const auto& set: sets) {
        sum += set.m_pieces;
    }
    
    cout << "Average part count for " << sets.size() << " sets: " << sum / sets.size() << endl;
}


void printPriceInfo(vector<LegoSet> sets) {
    int sum{};
    int maxIndex{};
    int minIndex{};
    for (size_t i = 1; i < sets.size(); ++i) {
        sum += sets[i].m_price;
        maxIndex = (sets[maxIndex].m_price > sets[i].m_price) ? maxIndex : i;
        minIndex = (sets[minIndex].m_price < sets[i].m_price) ? minIndex : i;
    }

    cout << "Average price information for " << sets.size() << " sets: $" << sum / sets.size() << "\n\n";
    cout << "Set with the minimum price:\n";
    printSetInfo(sets[minIndex]);
    cout << endl;
    cout << "Set with the maximum price:\n";
    printSetInfo(sets[maxIndex]);
}


void printMinifigInfo(vector<LegoSet> sets) {
    int sum{};
    int maxIndex{};
    for (size_t i = 1; i < sets.size(); ++i) {
        sum += sets[i].m_price;
        maxIndex = (sets[maxIndex].m_minifigs > sets[i].m_minifigs) ? maxIndex : i;
    }

    cout << "Average number of minifigures: " << sum / sets.size() << "\n";
    cout << "Set with the most minifigures:\n";
    printSetInfo(sets[maxIndex]);
}


void printOneOfEverything(vector<LegoSet> sets) {
    int costSum{};
    int pieceSum{};
    int minifigSum{};
    for (const auto& set: sets) {
        costSum += set.m_price;
        pieceSum += set.m_pieces;
        minifigSum += set.m_minifigs;
    }
    cout << "If you bought one of everything...\n";
    cout << "It would cost $" << costSum << endl;
    cout << "You would have " << pieceSum << " pieces in your collection\n";
    cout << "You would have an army of " << minifigSum << " minifigures!" << endl;
}