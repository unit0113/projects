#include "helpers.h"
#include "math.h"

// Convert image to grayscale
void grayscale(int height, int width, RGBTRIPLE image[height][width])
{
    
    int average;
    
    // Loop by row, then by column
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {

            // Determine the average value
            average = round((image[i][j].rgbtRed + image[i][j].rgbtGreen + image[i][j].rgbtBlue) / 3.0);
            //average = round(average);

            // Set all colors equal to average
            image[i][j].rgbtRed = average;
            image[i][j].rgbtGreen = average;
            image[i][j].rgbtBlue = average;
        }
    }

    return;
}

// Convert image to sepia
void sepia(int height, int width, RGBTRIPLE image[height][width])
{

    int sepiaRed, sepiaGreen, sepiaBlue, originalRed, originalGreen, originalBlue;
    
    // Loop by row, then by column
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            
            // Pull original colors
            originalRed = image[i][j].rgbtRed;
            originalGreen = image[i][j].rgbtGreen;
            originalBlue = image[i][j].rgbtBlue;
            
            // Convert to Sepia and round
            // Red
            sepiaRed = round(.393 * originalRed + .769 * originalGreen + .189 * originalBlue);
            if (sepiaRed > 255)
            {
                sepiaRed = 255;
            }
            
            // Green
            sepiaGreen = round(.349 * originalRed + .686 * originalGreen + .168 * originalBlue);
            if (sepiaGreen > 255)
            {
                sepiaGreen = 255;
            }
            
            // Blue
            sepiaBlue = round(.272 * originalRed + .534 * originalGreen + .131 * originalBlue);
            if (sepiaBlue > 255)
            {
                sepiaBlue = 255;
            }
            
            //Return edited Pixels
            image[i][j].rgbtRed = sepiaRed;
            image[i][j].rgbtGreen = sepiaGreen;
            image[i][j].rgbtBlue = sepiaBlue;

        }
    }

    return;
}

// Reflect image horizontally
void reflect(int height, int width, RGBTRIPLE image[height][width])
{
    
    RGBTRIPLE buffer[width];
    
    // Loop by row, then by column
    for (int i = 0; i < height; i++)
    {
        
        for (int j = 0; j < width; j++)
        {
            
            // Fill buffer from end
            buffer[(width - j - 1)] = image[i][j];
            
        }
        
        // Replace row with buffer
        for (int k = 0; k < width; k++)
        {
            image[i][k] = buffer[k];
        }
    }
    
    return;
}

// Blur image
void blur(int height, int width, RGBTRIPLE image[height][width])
{
    
    int aRed, aGreen, aBlue, pixels;
    RGBTRIPLE buffer[height][width];
    
    // Build buffer
    for (int i = 0; i < height; i++)
    {
        
        for (int j = 0; j < width; j++)
        {
            aRed = 0;
            aGreen = 0;
            aBlue = 0;
            pixels = 0;
            
            // Run through surrounding pixels
            for (int a = i - 1; a <= i + 1; a++)
            {
                for (int b = j - 1; b <= j + 1; b++)
                {
                    if (a < 0 || a > height - 1 || b < 0 || b > width - 1)
                    {
                        continue;
                    }
                    else
                    {
                        // Update totals
                        aRed += image[a][b].rgbtRed;
                        aBlue += image[a][b].rgbtBlue;
                        aGreen += image[a][b].rgbtGreen;
                        pixels++;
                    }
                }
            }
            
            // Update buffer
            aRed = round(aRed / (float)pixels);
            aGreen = round(aGreen / (float)pixels);
            aBlue = round(aBlue / (float)pixels);
            buffer[i][j].rgbtRed = aRed;
            buffer[i][j].rgbtGreen = aGreen;
            buffer[i][j].rgbtBlue = aBlue;
        }
    
    }
    
    // Update image
    for (int i = 0; i < height; i++)
    {
        
        for (int j = 0; j < width; j++)
        {
            image[i][j] = buffer[i][j];
        }
        
    }
    
    return;
}
