#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
using namespace std;

int main() {
   
    // Get Data title
    cout << "Enter a title for the data: ";
    string dataTitle;
    getline(cin, dataTitle);
    cout << "You entered: " << dataTitle << endl;

    // Get Col 1 title
    cout << "Enter the column 1 header: ";
    string colTitle1;
    getline(cin, colTitle1);
    cout << "You entered: " << colTitle1 << endl;

    // Get Col 2 title
    cout << "Enter the column 2 header: ";
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
        cout << "Enter a data point (-1 to stop input): ";
        getline(cin, dataPoint);

        // Break if -1
        if (dataPoint == "-1") {
            break;
        }

        // Check for correct num commas
        if (commaCount = count(dataPoint.begin(), dataPoint.end(), ',') == 0) {
            cout << "Error: No comma in string.\n" << endl;
            continue;
        } else if (commaCount > 1) {
            cout << "Error: Too many commas in input.\n" << endl;
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
            cout << "Error: Comma not followed by an integer.\n" << endl;
            continue;
        }

        // Good data, add to vectors
        strData.push_back(tempStr);
        intData.push_back(temptInt);
        
        // Output captured data
        cout << "Data string: " << tempStr << endl;
        cout << "Data integer: " << temptInt << endl;

    }









}