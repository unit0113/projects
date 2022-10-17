#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <initializer_list>
#include <stdexcept>
using namespace std;

string getStringFromBinaryFile(ifstream& file);

class Weapon {
	string m_name;
	int m_power;
	float m_power_consumption;

	public:
	Weapon(ifstream& file) {
		m_name = getStringFromBinaryFile(file);

		// Import numerics
		file.read(reinterpret_cast<char*>(&m_power), sizeof(m_power));
		file.read(reinterpret_cast<char*>(&m_power_consumption), sizeof(m_power_consumption));
	}

	void print() const {
		cout << m_name << ", " << m_power << ", " << m_power_consumption << endl;
	}

	int getPower() const {
		return m_power;
	}
};


class Ship {
	string m_name;
	string m_class;
	short m_length;
	int m_shield;
	float m_warp;
	vector<Weapon> m_weapons;

	public:
	Ship(ifstream& file) {
		m_name = getStringFromBinaryFile(file);
		m_class = getStringFromBinaryFile(file);
		file.read(reinterpret_cast<char*>(&m_length), sizeof(m_length));
		file.read(reinterpret_cast<char*>(&m_shield), sizeof(m_shield));
		file.read(reinterpret_cast<char*>(&m_warp), sizeof(m_warp));

		// Import weapons
		int weaponCount;
		file.read(reinterpret_cast<char*>(&weaponCount), sizeof(weaponCount));
		for (int i{}; i < weaponCount; ++i) {
			m_weapons.push_back(Weapon(file));
		}
	}

	void print() const {
		cout << "Name: " << m_name << endl;
		cout << "Class: " << m_class << endl;
		cout << "Length: " << m_length << endl;
		cout << "Shield capacity: " << m_shield << endl;
		cout << "Maximum Warp: " << m_warp << endl;

		cout << "Armaments:\n";
		if (m_weapons.size() == 0) {
			cout << "Unarmed\n";
		} else {
			int firepower{};
			for (const auto& weapon: m_weapons) {
				weapon.print();
				firepower += weapon.getPower();
			}
			cout << "Total firepower: " << firepower << endl;
		}
	}

	int getStrongestWeapon() const {
		int strongest{};
		for (const auto& weapon: m_weapons) {
			if (weapon.getPower() > strongest) {
				strongest = weapon.getPower();
			}
		}
		return strongest;
	}

	int getFirepower() const {
		int firepower{};
		for (const auto& weapon: m_weapons) {
			firepower += weapon.getPower();
		}
		return firepower;
	}

};


vector<Ship> loadFiles(std::initializer_list<string> args);
void printAllShips(const vector<Ship>& ships);
void printStrongestWeapon(const vector<Ship>& ships);
void printHighestFirepower(const vector<Ship>& ships);
void printWeakestShip(const vector<Ship>& ships);
void printUnarmed(const vector<Ship>& ships);

int main()
{
	cout << "Which file(s) to open?\n";
	cout << "1. friendlyships.shp" << endl;
	cout << "2. enemyships.shp" << endl;
	cout << "3. Both files" << endl;
	int option;
	cin >> option;

	// Load files
	vector<Ship> ships;
	switch (option) {
		case 1:
			ships = loadFiles({"friendlyships.shp"});
			break;
		case 2:
			ships = loadFiles({"enemyships.shp"});
			break;
		case 3:
			ships = loadFiles({"friendlyships.shp", "enemyships.shp"});
			break;
		default:
			throw std::runtime_error("Invalid Input");
            break;
	}

	cout << "1. Print all ships" << endl;
	cout << "2. Starship with the strongest weapon" << endl;
	cout << "3. Strongest starship overall" << endl;
	cout << "4. Weakest ship (ignoring unarmed)" << endl;
	cout << "5. Unarmed ships" << endl;
	
	cin >> option;
	
	switch (option) {
		case 1:
			printAllShips(ships);
			break;
		case 2:
			printStrongestWeapon(ships);
			break;
		case 3:
			printHighestFirepower(ships);
			break;
		case 4:
			printWeakestShip(ships);
			break;
		case 5:
			printUnarmed(ships);
			break;
		default:
			throw std::runtime_error("Invalid Input");
            break;
	}
	
   return 0;
}


vector<Ship> loadFiles(std::initializer_list<string> args) {
	ifstream file;
	vector<Ship> ships;
	int shipCount;

	// Loop through files
	for (const auto& fileName: args) {
		file.open(fileName, ios_base::binary);
		file.read(reinterpret_cast<char*>(&shipCount), sizeof(shipCount));

		// Parse all ships in file
		for (int i{}; i < shipCount; ++i) {
			ships.push_back(Ship(file));
		}

		file.close();
	}

	return ships;
}


string getStringFromBinaryFile(ifstream& file) {
	string result;
	int length;
	file.read(reinterpret_cast<char*>(&length), sizeof(length));
	char* temp = new char[length];
	file.read(temp, length);
	result = temp;
	delete[] temp;
	
	return result;
}


void printAllShips(const vector<Ship>& ships) {
	for (const auto& ship: ships) {
		ship.print();
		cout << endl;
	}
}


void printStrongestWeapon(const vector<Ship>& ships) {
	int strongestWeapon = ships[0].getStrongestWeapon();
	int strongestWeaponIndex{};

	for (size_t i = 1; i < ships.size(); ++i) { 
		if (ships[i].getStrongestWeapon() > strongestWeapon) {
			strongestWeapon = ships[i].getFirepower();
			strongestWeaponIndex = i;
		}
	}
	ships[strongestWeaponIndex].print();

}


void printHighestFirepower(const vector<Ship>& ships) {
	int maxFirePower = ships[0].getFirepower();
	int maxFirePowerIndex{};

	for (size_t i = 1; i < ships.size(); ++i) { 
		if (ships[i].getFirepower() > maxFirePower) {
			maxFirePower = ships[i].getFirepower();
			maxFirePowerIndex = i;
		}
	}
	ships[maxFirePowerIndex].print();
}


void printWeakestShip(const vector<Ship>& ships) {
	int minFirePower = ships[0].getFirepower();
	int minFirePowerIndex{};

	for (size_t i = 1; i < ships.size(); ++i) { 
		if (ships[i].getFirepower() != 0 && ships[i].getFirepower() < minFirePower) {
			minFirePower = ships[i].getFirepower();
			minFirePowerIndex = i;
		}
	}
	ships[minFirePowerIndex].print();
}


void printUnarmed(const vector<Ship>& ships) {
	for (const auto& ship: ships) {
		if (ship.getFirepower() == 0) {
			ship.print();
			cout << endl;
		}
	}
}