#include <iostream>
#include<unistd.h>
#include <algorithm>

int main(int argc, char** argv) {
    // Create pipes
    int parentToChild1Pipe[2];
    pipe(parentToChild1Pipe);
    int child1ToParent[2];
    pipe(child1ToParent);
    int child1ToChild2[2];
    pipe(child1ToChild2);
    int child2ToParent[2];
    pipe(child2ToParent);

    // Instansiate vars
    int nums[5];
    int median;

    // Create children
    int pid = fork();

    if (pid == 0) {
        close(parentToChild1Pipe[1]);
        close(child1ToParent[0]);
        read(parentToChild1Pipe[0], &nums, 5 * sizeof(int));

        // Create grandchild(?)
        pid = fork();
        if (pid == 0) {
            close(child1ToChild2[1]);
            close(child2ToParent[0]);
            read(child1ToChild2[0], &nums, 5 * sizeof(int));
            median = nums[2];
            write(child2ToParent[1], &median, sizeof(int));
        }

        // Continue child process
        std::sort(std::begin(nums), std::end(nums));
        write(child1ToParent[1], &nums, 5 * sizeof(int));
        write(child1ToChild2[1], &nums, 5 * sizeof(int));

        return 0;
    }

    // Parent process
    close(parentToChild1Pipe[0]);
    close(child1ToParent[1]);
    close(child2ToParent[1]);

    // Store args
    for (int i=0; i<5; ++i) {
        nums[i] = atoi(argv[i+1]);
    }
    write(parentToChild1Pipe[1], &nums, 5 * sizeof(int));
    read(child1ToParent[0], &nums, 5 * sizeof(int));

    std::cout << "Sorted list of ints: ";
    for (int i=0; i<5; ++i) {
        std::cout << nums[i] << "  ";
    }
    std::cout << std::endl;

    read(child2ToParent[0], &median, sizeof(int));
    std::cout << "Median: " << median << std::endl;
    
    return 0;
}