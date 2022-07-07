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

// Detect edges
void edges(int height, int width, RGBTRIPLE image[height][width])
{
    
    // G(X) Matrix
    int X[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    
    // G(Y) Matrix
    int Y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} }; 
    
    // Variables
    int xRed, xGreen, xBlue, yRed, yGreen, yBlue, gRed, gGreen, gBlue;
    RGBTRIPLE buffer[height][width];
    
    // Move through each pixel
    for (int i = 0; i < height; i++)
    {
        
        for (int j = 0; j < width; j++)
        {
            
            // Reset values
            xRed = xGreen = xBlue = yRed = yGreen = yBlue = 0;
            
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
                        // Update G(X) and G(Y) Matrices
                        xRed += X[a + 1 - i][b + 1 - j] * image[a][b].rgbtRed;
                        xGreen += X[a + 1 - i][b + 1 - j] * image[a][b].rgbtGreen;
                        xBlue += X[a + 1 - i][b + 1 - j] * image[a][b].rgbtBlue;
                        yRed += Y[a + 1 - i][b + 1 - j] * image[a][b].rgbtRed;
                        yGreen += Y[a + 1 - i][b + 1 - j] * image[a][b].rgbtGreen;
                        yBlue += Y[a + 1 - i][b + 1 - j] * image[a][b].rgbtBlue;
                    }
                }
            }
            
            // Calulate final colors
            gRed = round(sqrt(pow(xRed, 2.0) + pow(yRed, 2.0)));
            if (gRed > 255)
            {
                gRed = 255;
            }
            gGreen = round(sqrt(pow(xGreen, 2.0) + pow(yGreen, 2.0)));
            if (gGreen > 255)
            {
                gGreen = 255;
            }
            gBlue = round(sqrt(pow(xBlue, 2.0) + pow(yBlue, 2.0)));
            if (gBlue > 255)
            {
                gBlue = 255;
            }
            
            // Update buffer
            buffer[i][j].rgbtRed = gRed;
            buffer[i][j].rgbtGreen = gGreen;
            buffer[i][j].rgbtBlue = gBlue;
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
