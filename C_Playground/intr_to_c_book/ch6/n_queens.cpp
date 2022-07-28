#include <iostream>
#include <chrono>
 
#define N 8

int solve_nq(int board[N][N], int col);
int is_safe(int row, int col, int board[N][N]);
void print_board(int board[N][N]);


int main() {

    auto start = std::chrono::high_resolution_clock::now();

    int board[N][N];
    
    // Initialize board
    for (int row = 0; row < N; row++)
        for (int col = 0; col < N; col++)
            board[row][col] = 0;

    // Initialize knight to top left block
    board[0][0] = 0;

    // Start from 0,0 and explore all tours
    if (solve_nq(board, 0) == 0) {
        std::cout << "Solution does not exist\n";
    } else {
        print_board(board);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;
 
    return 0;
}


int solve_nq(int board[N][N], int col) {
    // Base case, all queens placed
    if (col >= N) {
        return true;
    }

    // Try all rows
    for (int i = 0; i < N; i++) {
        if (is_safe(i, col, board)) {
            /* Place this queen in board[i][col] */
            board[i][col] = 1;
  
            /* recur to place rest of the queens */
            if (solve_nq(board, col + 1))
                return true;
  
            // Backtrack
            board[i][col] = 0;
        }
    }
    return false;
}


int is_safe(int row, int col, int board[N][N]) {
    int i, j;
    // Check row to the left
    for (i = 0; i < col; i++) {
        if (board[row][i]) {
            return false;
        }
    }

    // Check upper left diag
    for (i = row, j = col; i >= 0 && j >= 0; i--, j--) {
        if (board[i][j]) {
            return false;
        }
    }

    // Check lower left diag
    for (i = row, j = col; i < N && j >= 0; i++, j--) {
        if (board[i][j]) {
            return false;
        }
    }

    return true;
}


void print_board(int board[N][N]) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            std::cout << " " << board[row][col] << " ";
        std::cout << std::endl;
    }
}