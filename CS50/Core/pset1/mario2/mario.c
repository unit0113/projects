#include <stdio.h>
#include <cs50.h>

int main(void)
{
    printf("Welcome to Mario level maker 1.0!\n");
    int height;

    //Get int for height, 1-8 are valid
    do
    {
        height = get_int("How high should we go (1-8)?\n");
    }
    while (height > 8 || height < 1);

    //Build the pyramid
    for (int i = 0; i < height; i++)
    {
        for (int k = height - 1 - i; k > 0; k--) //inner look for spaces
        {
            printf(" ");
        }

        for (int j = 0; j < i + 1; j++) //inner loop for #
        {
            printf("#");
        }
        
        printf("  "); //space in the middle
        
        for (int l = 0; l < i + 1; l++) //inner loop for second set of #
        {
            printf("#");
        }
        
        printf("\n");
    }

}
