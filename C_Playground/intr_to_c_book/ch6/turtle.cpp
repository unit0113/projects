#include <iostream>


void pen_up(bool &pen);
void pen_down(bool &pen);
void turn_left(int &direction);
void turn_right(int &direction);
void move(int arr1[][20], int row, int col, int dist);
void print();


enum directions{up, right, down, left};

int main() {

    const int grid_size = 20;
    int grid[grid_size][grid_size] = {0};
    int row = 0;
    int col = 0;
    int direction = up;
    bool draw = true;




    return 0;
}

void pen_up(bool &pen) {
    pen = false;
}


void pen_down(bool &pen) {
    pen = true;
}


void turn_left(int &direction) {
    direction = (direction - 1) % 4;
    if (direction < 0) {
        direction += 4;
    }
}


void turn_right(int &direction) {
    direction = (direction + 1) % 4;
}