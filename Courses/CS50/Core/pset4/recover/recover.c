#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BLOCK 512

int main(int argc, char *argv[])
{

    // Check input
    if (argc != 2)
    {
        printf("Usage: ./recover image\n");
        return 1;
    }

    // Remember filename
    char *infile = argv[1];

    // Open input file
    FILE *input = fopen(infile, "r");
    if (input == NULL)
    {
        fprintf(stderr, "Could not open %s\n", infile);
        return 2;
    }

    // Declare stuff
    int counter = 0;
    uint8_t buffer[BLOCK];
    char filename[10];
    FILE *jpeg = NULL;

    // Big loop
    while (fread(buffer, BLOCK, 1, input)) // Maybe & in front of buffer?
    {
        // Check start of block
        if ((buffer[0] == 0xff) && (buffer[1] == 0xd8) && (buffer[2] == 0xff) && ((buffer[3] & 0xf0) == 0xe0))
        {
            // Close previous file
            if (counter > 0)
            {
                fclose(jpeg);
            }

            // Create and open new .jpeg file
            sprintf(filename, "%03i.jpg", counter); // Sets string filename to "000.jpg, 001.jpg"
            jpeg = fopen(filename, "w");
            counter++;

            //Add buffer to file
            fwrite(buffer, BLOCK, 1, jpeg);
        }

        else
        {
            if (counter > 0) // No adding data to null JPG
            {
                // Write to file
                fwrite(buffer, BLOCK, 1, jpeg);
            }
        }

    }

    // Close final files
    fclose(jpeg);
    fclose(input);
    return 0;

}