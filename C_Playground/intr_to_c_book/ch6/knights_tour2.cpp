#include <iostream>
#include <iomanip>
#include <chrono>
 
#define N 8

int solve_kt(int x, int y, int board[N][N], int move_count);
int isSafe(int x, int y, int board[N][N]);
void print_board(int board[N][N]);


int main() {

    auto start = std::chrono::high_resolution_clock::now();

    int board[N][N];
    
    // Initialize board
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
            board[row][col] = -1;

    // Initialize knight to top left block
    board[0][0] = 0;

    // Start from 0,0 and explore all tours
    if (solve_kt(0, 0, board, 1) == 0) {
        std::cout << "Solution does not exist";
    } else {
        print_board(board);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;
 
    return 0;
}


int solve_kt(int x, int y, int board[N][N], int move_count) {
    static const int num_pos_moves = 8;
    static const int xMove[num_pos_moves] = { 2, 1, -1, -2, -2, -1, 1, 2 };
    static const int yMove[num_pos_moves] = { 1, 2, 2, 1, -1, -2, -2, -1 };

    int next_x, next_y;

    // If out of moves
    if (move_count == N * N)
        return 1;

    // Loop through all possible moves
    for (size_t move = 0; move < num_pos_moves; move++) {
        next_x = x + xMove[move];
        next_y = y + yMove[move];

        // Check if valid move
        if (isSafe(next_x, next_y, board)) {
            board[next_x][next_y] = move_count;

            if (solve_kt(next_x, next_y, board, move_count+1) == 1) {
                return 1;
            } else {
               // backtracking
                board[next_x][next_y] = -1;
            }
        }
    }
    return 0;
}


int isSafe(int x, int y, int board[N][N]) {
    return ((x >= 0) && (x < N) && (y >= 0) && (y < N) && (board[x][y] == -1));
}


void print_board(int board[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            std::cout << " " << std::setw(2) << board[row][col] << " ";
        std::cout << std::endl;
    }
}