// Implements a dictionary's functionality

#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

#include "dictionary.h"

// Represents a node in a hash table
typedef struct node
{
    char word[LENGTH + 1];
    struct node *next;
}
node;

// Number of buckets in hash table
const unsigned int N = 1000;

// Hash table
node *table[N];

// Number of Loaded Words
unsigned int total = 0;

// Returns true if word is in dictionary, else false
bool check(const char *word)
{
    // Hash it
    int index = hash(word);

    // Search Hash
    node *cursor = table[index];
    while (cursor != NULL)
    {
        if (strcasecmp(cursor->word, word) == 0)
        {
            return true;
        }
        
        else
        {
            cursor = cursor->next;
        }
    }

    return false;
}

// Hashes word to a number
unsigned int hash(const char *word)
{
    // Hash it
    int sum = 0;
    for (int i = 0; i < strlen(word); i++)
    {
        sum += N * tolower(word[i]);
    }
    return (sum % N);
}

// Loads dictionary into memory, returning true if successful, else false
bool load(const char *dictionary)
{
    // Open File
    FILE *dict = fopen(dictionary, "r");
    if (dict == NULL)
    {
        return false;
    }

    // Read File
    char word[LENGTH + 1];
    while (fscanf(dict, "%s", word) != EOF)
    {
        // Create Nodes
        node *newnode = malloc(sizeof(node));
        if (newnode == NULL)
        {
            return false;
        }

        // Copy Word to Node
        strcpy(newnode->word, word);
        newnode->next = NULL;

        // Hash it
        int index = hash(word);

        if (table[index] == NULL)
        {
            table[index] = newnode;
        }
        else
        {
            newnode->next = table[index];
            table[index] = newnode;
        }
        total ++;
    }
    fclose(dict);
    return true;
}

// Returns number of words in dictionary if loaded, else 0 if not yet loaded
unsigned int size(void)
{
    // TODO
    return total;
}

// Unloads dictionary from memory, returning true if successful, else false
bool unload(void)
{
    // Temp Pointers
    node *cursor = NULL;
    node *tmp = NULL;

    // Loop for Table
    for (int i = 0; i < N; i++)
    {
        cursor = table[i];

        // Loop in Hash Table
        while (cursor != NULL)
        {
            tmp = cursor;
            cursor = cursor->next;
            free(tmp);
        }

    }

    return true;
}
