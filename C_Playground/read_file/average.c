#include <stdio.h>


int main() {
    FILE *file;
    if ((file = fopen("elephant_seal_data.txt", "r")) == NULL) {
        printf("Error opening file");
        return 1;
    }

    int counter = 0;
    int sum = 0;
    int weights[1000];

    while(fscanf(file, "%d", &weights[counter]) != EOF) {
        sum += weights[counter++];
    }

    printf("Average weight: %.3lf", (double)sum / counter);
    fclose(file);
}