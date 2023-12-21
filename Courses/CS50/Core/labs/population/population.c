#include <cs50.h>
#include <stdio.h>

int main(void)
{
    // Prompt for start size
    int start;

    do
    {
        start = get_int("Start size: ");
    }
    while (start < 9);
    
    // Prompt for end size
    int end;

    do
    {
        end = get_int("End size: ");
    }
    while (end < start);
    
    // Calculate number of years until we reach threshold
    int current = start;
    int years = 0;
    int growth;
    int death;
    
    while (current < end)
    {
        growth = current / 3;
        death = current / 4;
        current = current + growth - death;
        years++;
    }

    // Print number of years
    printf("Years: %i\n", years);
}