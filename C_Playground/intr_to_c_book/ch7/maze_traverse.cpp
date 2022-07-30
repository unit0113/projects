#include <iostream>
#include <vector>
#include <queue>
#include <chrono>


#define MAZE_SIZE 12


struct PathlessPoint {
    int x, y;
};

struct Point {
    int x, y;
    std::vector<PathlessPoint> path;
};


void print_maze(char board[MAZE_SIZE][MAZE_SIZE], std::vector<PathlessPoint> path);
std::vector<Point> get_neighbors(Point p1, char maze[MAZE_SIZE][MAZE_SIZE]);
std::vector<PathlessPoint> bfs(Point start_point, PathlessPoint end_point, char maze[MAZE_SIZE][MAZE_SIZE]);


int main() {

    auto start = std::chrono::high_resolution_clock::now();

    char maze[MAZE_SIZE][MAZE_SIZE] = {{'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'},
                                       {'#', '.', '.', '.', '#', '.', '.', '.', '.', '.', '.', '#'},
                                       {'.', '.', '#', '.', '#', '.', '#', '#', '#', '#', '.', '#'},
                                       {'#', '#', '#', '.', '#', '.', '.', '.', '.', '#', '.', '#'},
                                       {'#', '.', '.', '.', '.', '#', '#', '#', '.', '#', '.', '.'},
                                       {'#', '#', '#', '#', '.', '#', '.', '#', '.', '#', '.', '#'},
                                       {'#', '.', '.', '#', '.', '#', '.', '#', '.', '#', '.', '#'},
                                       {'#', '#', '.', '#', '.', '#', '.', '#', '.', '#', '.', '#'},
                                       {'#', '.', '.', '.', '.', '.', '.', '.', '.', '#', '.', '#'},
                                       {'#', '#', '#', '#', '#', '#', '.', '#', '#', '#', '.', '#'},
                                       {'#', '.', '.', '.', '.', '.', '.', '#', '.', '.', '.', '#'},
                                       {'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'}};

    // Initial points
    Point start_point {2, 0};
    start_point.path.push_back(PathlessPoint {start_point.x, start_point.y});
    PathlessPoint end_point {4, 11};
    maze[start_point.x][start_point.y] = ' ';

    std::vector<PathlessPoint> path = bfs(start_point, end_point, maze);

    print_maze(maze, path);



    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}


void print_maze(char maze[MAZE_SIZE][MAZE_SIZE], std::vector<PathlessPoint> path) {

    for (PathlessPoint pt: path) {
        maze[pt.x][pt.y] = 'X';
    }

    for (int row = 0; row < MAZE_SIZE; row++) {
        for (int col = 0; col < MAZE_SIZE; col++)
            std::cout << " " << maze[row][col] << " ";
        std::cout << std::endl;
    }

}


std::vector<Point> get_neighbors(Point p1, char maze[MAZE_SIZE][MAZE_SIZE]) {
    std::vector<Point> neighbors;

    static const int num_directions = 4;
    static const int xMove[num_directions] = { 1, -1, 0, 0 };
    static const int yMove[num_directions] = { 0, 0, 1, -1 };

    int new_x = 0;
    int new_y = 0;
    for (size_t i = 0; i < num_directions; i++) {
        new_x = p1.x + xMove[i];
        new_y = p1.y + yMove[i];

        if ((new_x < 0) 
            || (new_x >= MAZE_SIZE)
            || (new_y < 0)
            || (new_y >= MAZE_SIZE)
            || (maze[new_x][new_y] != '.')) {
                continue;
        } else {
            neighbors.push_back(Point {new_x, new_y});
        }
    }

    return neighbors;

}

std::vector<PathlessPoint> bfs(Point start_point, PathlessPoint end_point, char maze[MAZE_SIZE][MAZE_SIZE]) {
    // Initialize queue
    std::queue<Point> que;
    que.push(start_point);

    Point working_point;

    while (!que.empty()) {
        working_point = que.front();
        que.pop();

        for (Point neighbor: get_neighbors(working_point, maze)) {

            std::cout << "Exploring (" << neighbor.x << ", " << neighbor.y << ")\n";

            // If end point found
            if (neighbor.x == end_point.x && neighbor.y == end_point.y) {
                working_point.path.push_back(end_point);
                return working_point.path;
            } else {
                maze[neighbor.x][neighbor.y] = ' ';
                neighbor.path = working_point.path;
                neighbor.path.push_back(PathlessPoint {neighbor.x, neighbor.y});
                que.push(neighbor);
            }
        }

    }
}