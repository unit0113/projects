/*
 * bf_vuln.c
 * This program is vulnerable to buffer overflow
 * and is solely for demonstrating purposes
 * Author: Nik Alleyne < nikalleyne at gmail dot com >
 * blog: http://securitynik.blogspot.com
 * Date: 2017-01-01
 */

#include <stdio.h>
#include <unistd.h>

// This will be our arbitrary code
// Nothing malicious just fun
void arbitrary_code()
{
    printf(" Now you know I should not be seen ! \n");
    printf(" But I am. Don't believe me just watch \n");
}


//Our vulnerable function
void get_input()
{
    //declaring our buffer size of 8 bytes
    char buffer[8];
    //read our input. This 'gets' function is where our problem really lies
    gets(buffer);
    //print the output back to the screen. Simply echo our input back to the screen
    puts(buffer);
}

int  benign_function()
{
	int x, y, z;
	x=3;
	y=4;
	z=x+y;
        return z;
}
int main()
{
    int a;
    a = benign_function();
    get_input();
    return 0;
}
/* Note: Nowhere in the above code did we call 'arbitrary_code'
 * However, we will see shortly that we can exectue this code
 * by using its memory location
 */