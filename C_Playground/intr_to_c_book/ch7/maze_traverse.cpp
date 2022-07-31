#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <iterator>
#include <array>
#include <chrono>


#define MAZE_SIZE 12


struct Point {
    int x, y;
    std::vector<Point> path;
};


void print_maze(std::vector<std::vector<char>> maze, std::vector<Point> const path);
std::vector<Point> get_neighbors(const Point p1, std::vector<std::vector<char>> const maze);
std::vector<Point> bfs(const Point start_point, const Point end_point, std::vector<std::vector<char>> &maze);
std::vector<Point> dfs(const Point start_point, const Point end_point, std::vector<std::vector<char>> &maze);


int main() {

    std::vector<std::vector<char>> maze {{'#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#', '#'},
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

    std::vector<std::vector<char>> maze2 = maze;

    // Initial points
    Point start_point {2, 0};
    start_point.path.push_back(Point {start_point.x, start_point.y});
    Point end_point {4, 11};
    maze[start_point.x][start_point.y] = ' ';
    maze2[start_point.x][start_point.y] = ' ';

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Point> path_bfs = bfs(start_point, end_point, maze);

    if (path_bfs.empty()) {
        std::cout << "No path found" << std::endl;
    } else {
        print_maze(maze, path_bfs);
    }  

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "BFS Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;


    start = std::chrono::high_resolution_clock::now();

    std::vector<Point> path_dfs = dfs(start_point, end_point, maze2);

    if (path_dfs.empty()) {
        std::cout << "No path found" << std::endl;
    } else {
        print_maze(maze2, path_dfs);
    }  

    end = std::chrono::high_resolution_clock::now();
    std::cout << "DFS Elapsed time: " << std::chrono::duration<double>(end - start).count() << std::endl;

    return 0;
}


void print_maze(std::vector<std::vector<char>> maze, std::vector<Point> const path) {

    for (Point pt: path) {
        maze[pt.x][pt.y] = 'X';
    }

    for (int row = 0; row < MAZE_SIZE; row++) {
        for (int col = 0; col < MAZE_SIZE; col++)
            std::cout << " " << maze[row][col] << " ";
        std::cout << std::endl;
    }

}


std::vector<Point> get_neighbors(const Point p1, std::vector<std::vector<char>> const maze) {
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


std::vector<Point> bfs(const Point start_point, const Point end_point, std::vector<std::vector<char>> &maze) {

    std::cout << "Beginning breadth first search" << std::endl;

    // Initialize queue
    std::queue<Point> que;
    que.push(start_point);

    Point working_point;

    while (!que.empty()) {
        working_point = que.front();
        maze[working_point.x][working_point.y] = ' ';
        que.pop();

        for (Point neighbor: get_neighbors(working_point, maze)) {

            std::cout << "Exploring (" << neighbor.x << ", " << neighbor.y << ")\n";

            // If end point found
            if (neighbor.x == end_point.x && neighbor.y == end_point.y) {
                working_point.path.push_back(end_point);
                return working_point.path;
            } else {
                neighbor.path = working_point.path;
                neighbor.path.push_back(Point {neighbor.x, neighbor.y});
                que.push(neighbor);
            }
        }
    }
    std::vector<Point> no_path_vec;
    return no_path_vec;
}


std::vector<Point> dfs(const Point start_point, const Point end_point, std::vector<std::vector<char>> &maze) {

    std::cout << "Beginning depth first search" << std::endl;

    // Initialize stack
    std::vector<Point> stack;
    stack.push_back(start_point);

    Point working_point;

    while (!stack.empty()) {
        working_point = stack.back();
        maze[working_point.x][working_point.y] = ' ';
        stack.pop_back();

        for (Point neighbor: get_neighbors(working_point, maze)) {

            std::cout << "Exploring (" << neighbor.x << ", " << neighbor.y << ")\n";

            // If end point found
            if (neighbor.x == end_point.x && neighbor.y == end_point.y) {
                working_point.path.push_back(end_point);
                return working_point.path;
            } else {
                neighbor.path = working_point.path;
                neighbor.path.push_back(Point {neighbor.x, neighbor.y});
                stack.push_back(neighbor);
            }
        }
    }
    std::vector<Point> no_path_vec;
    return no_path_vec;
}