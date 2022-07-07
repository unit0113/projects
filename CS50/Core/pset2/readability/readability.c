#include <stdio.h>
#include <cs50.h>
#include <string.h>
#include <math.h>
#include <ctype.h>



int main(void)
{
    //Variables
    string text = get_string("Text: \n");
    float l = 0, w = 0, s = 0;
    bool wc = false;
    
    for (int i = 0; i < strlen(text); i++)
    {
        //Count letters
        if (islower(text[i]))
        {
            l++;
            wc = true;
        }
        else if (isupper(text[i]))
        {
            l++;
            wc = true;
        }

        //Count words
        else if (wc == true && isspace(text[i]))
        {
            w++;
            wc = false;
        }

        //Count sentences
        else if (text[i] == '.' || text[i] == '?' || text[i] == '!')
        {
            s++;
        }

    }
   
    //Math
    float L = (l / w) * 100;
    float S = (s / w) * 100;

    //Output result
    float grade = 0.0588 * L - 0.296 * S - 15.8;

    //Print grade
    if (grade < 1)
    {
        printf("Before Grade 1\n");
    }

    else if (grade > 16)
    {
        printf("Grade 16+\n");
    }

    else
    {
        int fgrade = round(grade);
        printf("Grade %i\n", fgrade);
    }
}