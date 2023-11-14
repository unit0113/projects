#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>

using namespace std;


string getBinary(char hex) {
    if (hex=='0')
        return "0000";
    if (hex=='1')
        return "0001";
    if (hex=='2')
        return "0010";
    if (hex=='3')
        return "0011";
    if (hex=='4')
        return "0100";
    if (hex=='5')
        return "0101";
    if (hex=='6')
        return "0110";
    if (hex=='7')
        return "0111";
    if (hex=='8')
        return "1000";
    if (hex=='9')
        return "1001";
    if ((hex=='a')||(hex=='A'))
        return "1010";
    if ((hex=='b')||(hex=='B'))
        return "1011";
    if ((hex=='c')||(hex=='C'))
        return "1100";
    if ((hex=='d')||(hex=='D'))
        return "1101";
    if ((hex=='e')||(hex=='E'))
        return "1110";
    if ((hex=='f')||(hex=='F'))
        return "1111";

}

int getTag(string addr, int tagSize)
{
    string tagBinary;
    int tag=0;
    string extra;
    int numHex = tagSize/4;
    int numExtra = tagSize % 4;

    int i;
    for  (i=0; i<numHex; i++)
        tagBinary += getBinary(addr[i+2]);

    if (numExtra > 0) {
        extra = getBinary(addr[i+2]);
        for (int j = 0; j<numExtra; j++)
            tagBinary+=extra[j];
    }
    int multiplier=1;
    for (i=tagSize-1;i>=0; i--) {
        if (tagBinary[i]=='1')
            tag+=multiplier;
        multiplier *= 2;
    }
    return tag;
}
int getSet(string addr, int tagsize, int setsize) {
    int set=0;
    string binaryAddress;
    string setBinary;

    for (int i=0; i<8; i++)
        binaryAddress+=getBinary(addr[i+2]);
    for (int i=0; i<setsize; i++)
        setBinary += binaryAddress[tagsize+i];

    //now turn setBinary into decimal
    int multiplier=1;
    for (int i=setsize-1; i>=0; i--) {
        if (setBinary[i]=='1')
            set+=multiplier;
        multiplier *=2;
    }
    return set;
}
bool checkCache(int set, int setSizeExp,  vector<vector<int> > &cache, int tag, int counter, string replacementStrategy) {
    if (!setSizeExp) //direct mapped
    {
        if (cache[set][0] == tag) {
            cache[set][1]=counter;
            return true;
        }
        else {
            cache[set][0]=tag;
            cache[set][1]=counter;
            return false;
        }
    }
    float setSize = pow(2, setSizeExp);
    int j=set*setSize;
    int emptySpot=-1;
    int smallestCounter=-1;
    int lineToReplace=-1;
    for (int i = 0; i<setSize; i++) {
        if (cache[i+j][0]==tag) {
            cache[i+j][1]=counter;
            return true;
        }
        else if (cache[i+j][0]==-1) {
            emptySpot = i+j;
        }
        else if (smallestCounter==-1) {
            smallestCounter = cache[i+j][1];
            lineToReplace=i+j;

        }
        else if(cache[i+j][1]<smallestCounter) {
            smallestCounter = cache[i+j][1];
            lineToReplace=i+j;
        }

    }
    //empty spot?
if (emptySpot!=-1)  //there was an empty spot, fill it
{
    cache[emptySpot][0]=tag;
    cache[emptySpot][1]=counter;
}
else //update entry with lowest counter
{
    cache[lineToReplace][0]=tag;
    if (replacementStrategy == "LRU")
        cache[lineToReplace][1]=counter;
}
   return false;

}

int main() {
    int cacheSizeExp;

    cout << "This is a rudimentary cache simulator.  It is your responsibility to ensure that the parameters you enter make sense" << endl;
    cout << "Cache size is an exponent of 2.  E.g. if the exponent is 3, the cache is 2 to the 3, or 8 bytes" << endl;
    cout << "Enter the exponent for the cache size" << endl;    
    cin >> cacheSizeExp;
 
    int lineSizeExp; //line size = 2^lineSizeExp, lineSizeExp=size of offset field
    cout << "Line size is an exponent of 2.  E.g. if the exponent is 3, the cache is 2 to the 3, or 8 bytes" << endl;
    cout << "Enter the exponent for the line size" << endl;    
    cin >> lineSizeExp;

    int numLinesExp = cacheSizeExp - lineSizeExp;
    //numLines = 2^numLinesExp

    int setSizeExp;  //zero for direct mapped (2^0 = 1 line/set), numLinesExp for fully associative (1 set)
    char fa, dm;
    int numLinesPerSet;

    cout << "Is the cache fully associative? Enter 'Y' or 'y' if yes, any other character if no " << endl; 
    cin >> fa;

    if ((fa=='y')||(fa=='Y'))
 	setSizeExp=numLinesExp;
    else {
        cout << "Is the cache direct mapped? Enter 'Y' or 'y' if yes, any other character if no " << endl; 
   	    cin >> dm;
            if ((dm=='y')||(dm=='Y'))
 	         setSizeExp=0;
            else {
	    	cout << "Enter '1' for 2 lines per set, '2' for 4 lines per set, '3' for 8 lines per set, or '4' for 16 lines per set." << endl;
	    	cin >> setSizeExp;
	    	if ((setSizeExp>4)||(setSizeExp<1))	{
			cout << "Try again. It's your responsibility to enter numbers that make sense" << endl;
                	return 0;
	    	}
        }
    }
    
    int numSetsExp = numLinesExp - setSizeExp; //set field size
    //zero for fully associative
    int tagsize = 32 - numSetsExp - lineSizeExp;
    int numLines = pow(2, numLinesExp);
    vector<vector<int> > cache(numLines);
    for (int i=0; i<numLines; i++) {
        //each line has three parameters: tag, set, access time
        //set all to -1 to start
        cache[i] = vector<int>(2);
        cache[i][0]=-1; //tag
        cache[i][1]=-1; //access counter
    }

    //string filename;
    //cout << "Enter filename " << endl;
    //cin >> filename;	

    vector<string> replacementStrategies = {"LRU", "FIFO"};
    vector<string> files = {"traces/gcc.trace", "traces/swim.trace"};
    for (string replacementStrategy: replacementStrategies) {
        for (string filename: files) {
            ifstream newfile(filename);
            string ls, addr, bytes;
            int counter = 0;
            bool hit;
            int numhits = 0;
            while (!newfile.eof()) {
                getline(newfile, ls,' ');
                getline(newfile, addr, ' ');
                int tag = getTag(addr, tagsize);
                int set;
                if (!numSetsExp)
                    //if numSetsExp=0, then number of sets = 1 (2^0=1), and it is fully associative
                    //there is only one set
                    set = 0;
                else
                    set = getSet(addr, tagsize, numSetsExp);
                //check for hit or miss
                if (checkCache(set, setSizeExp, cache, tag, counter, replacementStrategy)) {
                    numhits++;
                }

                getline(newfile, bytes);
                counter++;
            }
            float hitrate = (float) numhits/(float) counter;
            cout << "File: " << filename << "Replacement Strategy: " << replacementStrategy << ", Hits: " << numhits << ", Accesses: " << counter << ", Hit rate: " << hitrate << endl;
        }
        if (setSizeExp == 0)
            break;
    }
    return 0;
}
