#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
using namespace std;

int main() {
   
    // Get Data title
    cout << "Enter a title for the data:\n";
    string dataTitle;
    getline(cin, dataTitle);
    cout << "You entered: " << dataTitle << endl;

    // Get Col 1 title
    cout << "Enter the column 1 header:\n";
    string colTitle1;
    getline(cin, colTitle1);
    cout << "You entered: " << colTitle1 << endl;

    // Get Col 2 title
    cout << "Enter the column 2 header:\n";
    string colTitle2;
    getline(cin, colTitle2);
    cout << "You entered: " << colTitle2 << endl;

    // Temp data
    string dataPoint;
    size_t commaCount{ 0 };
    size_t commaIndex{ 0 };
    string tempStr;
    int temptInt;

    // Data vectors
    vector<string> strData;
    vector<int> intData;

    while (true) {
        // Get input
        cout << "Enter a data point (-1 to stop input):\n";
        getline(cin, dataPoint);

        // Break if -1
        if (dataPoint == "-1") {
            cout << endl;
            break;
        }

        // Check for correct num commas
        if ((commaCount = count(dataPoint.begin(), dataPoint.end(), ',')) == 0) {
            cout << "Error: No comma in string." << endl;
            continue;
        } else if (commaCount > 1) {
            cout << "Error: Too many commas in input." << endl;
            continue;
        }

        // Extract str
        commaIndex = dataPoint.find(',');
        tempStr = dataPoint.substr(0, commaIndex);

        // Extract int
        try {
            temptInt = stoi(dataPoint.substr(commaIndex + 1, dataPoint.size() - commaIndex));
        }
        catch (invalid_argument) {
            cout << "Error: Comma not followed by an integer." << endl;
            continue;
        }

        // Good data, add to vectors
        strData.push_back(tempStr);
        intData.push_back(temptInt);
        
        // Output captured data
        cout << "Data string: " << tempStr << endl;
        cout << "Data integer: " << temptInt << endl;

    }

    // Table 1
    // Table title
    cout << right << setw(33) << dataTitle << endl;

    // Col headers
    cout << left << setw(20) << colTitle1 << '|' << right << setw(23) << colTitle2 << endl;;

    // Divider
    cout << string(44, '-') << endl;

    // Data loop
    for (size_t i = 0; i < strData.size(); i++) {
        cout << left << setw(20) << strData[i] << '|' << right << setw(23) << intData[i] << endl;;
    }

    cout << endl;

    // Table 2
    for (size_t i = 0; i < strData.size(); i++) {
        cout << right << setw(20) << strData[i] << ' ' << string(intData[i], '*') << endl;;
    }

}