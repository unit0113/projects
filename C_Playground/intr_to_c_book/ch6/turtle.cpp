#include <iostream>
#include <vector>
#include <algorithm>

void print_orders();
void pen_up(bool &pen);
void pen_down(bool &pen);
void turn_left(int &direction);
void turn_right(int &direction);
void move(int arr[][50], int &row, int &col, int dist, bool pen, int &direction);
void print_grid(int arr[][50]);


enum directions{up, right, down, left};

int main() {

    const int grid_size = 50;
    int grid[grid_size][grid_size] = {0};
    int row = 0;
    int col = 0;
    int direction = up;
    bool draw = true;

    print_orders();
    int last_entered = 0;
    int dist = 0;
    while (last_entered != 9) {
        std::cout << "Enter a command:: ";
        std::cin >> last_entered;
        if (0 < last_entered && last_entered < 9) {
            switch (last_entered) {
                case 1:
                    pen_up(draw);
                    break;
                
                case 2:
                    pen_down(draw);
                    break;

                case 3:
                    turn_right(direction);
                    break;

                case 4:
                    turn_left(direction);
                    break;
                
                case 5:
                    std::cin >> dist;
                    move(grid, row, col, dist, draw, direction);
                    break;

                case 6:
                    print_grid(grid);
                    break;

                case 7:
                    std::cout << "Location: (" << row << ", " << col << "), ";
                    if (draw) {
                        std::cout << "Pen down, ";
                    } else {
                        std::cout << "Pen up, ";
                    }
                    if (direction == up) {
                        std::cout << "Direction: up";
                    } else if (direction == down) {
                        std::cout << "Direction: down";
                    } else if (direction == right) {
                        std::cout << "Direction: right";
                    }else if (direction == left) {
                        std::cout << "Direction: left";
                    }
                    std::cout << std::endl;
                    break;

                case 8:
                    print_orders();
                    break;

                default:
                    std::cout << "Invalid entry\n";
                    break;
            }
        }
    }

    return 0;
}


void print_orders() {
    std::cout << "Available options:" << std::endl;
    std::cout << "1: Pen up\n";
    std::cout << "2: Pen down\n";
    std::cout << "3: Turn right\n";
    std::cout << "4: Turn left\n";
    std::cout << "5 int: Move forward by int spaces\n";
    std::cout << "6: Print grid\n";
    std::cout << "7: Display turtle status\n";
    std::cout << "8: Display options again\n";
    std::cout << "9: End of data" << std::endl;
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


void move(int arr[][50], int &row, int &col, int dist, bool pen, int &direction) {
    size_t plus_minus = 1;
    if (direction == up || direction == left) {
        plus_minus *= -1;
    }

    if (pen) {
        for (size_t i = 0; i < dist; i++) {
            arr[row][col] = 1;
            if (direction == right || direction == left) {
                col += plus_minus;
                if (col < 0 || col > 49) {
                    col = std::clamp(col, 0, 49);
                    break;
                }
            } else {
                row += plus_minus;
                if (row < 0 || row > 49) {
                    row = std::clamp(row, 0, 49);
                    break;
                }
            }
        }

    } else {
        if (direction == up) {
            col = std::max(0, col-dist);
        } else if (direction == down) {
            col = std::min(49, col+dist);
        } else if (direction == right) {
            row = std::min(49, row+dist);
        } else if (direction == left) {
            row = std::max(0, row-dist);
        }
    }
}


void print_grid(int arr[][50]) {
    for (size_t row = 0; row < 50; row++) {
        for (size_t col = 0; col < 50; col++) {
            if (arr[row][col] == 1) {
                std::cout << '*';
            } else {
                std::cout << ' ';
            }
        }
        std::cout << std::endl;
    }
}
