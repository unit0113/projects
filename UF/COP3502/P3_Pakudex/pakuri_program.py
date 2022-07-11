from pakudex import pakudex


def print_menu():
    print()
    print('Pakudex Main Menu')
    print('-----------------')
    print('1. List Pakuri')
    print('2. Show Pakuri')
    print('3. Add Pakuri')
    print('4. Evolve Pakuri')
    print('5. Sort Pakuri')
    print('6. Exit')
    print()


def menu():
    print_menu()
    # Loop until valid input, print menu after each failure
    while True:
        try:
            selection = int(input('What would you like to do? '))
            if selection < 1 or selection > 6:
                raise ValueError
            else:
                break
        # Catch non int and ints outside of valid range
        except ValueError:
            print('Unrecognized menu selection!')
            print_menu()
            continue

    return selection


def print_pakus(pakus):
    if pakus:
        print('Pakuri In Pakudex:')
        for index, paku_name in enumerate(pakus):
            print(f'{index+1}. {paku_name}')

    # If no pakus
    else:
        print('No Pakuri in Pakudex yet!')


def show_paku(paku, paku_stats):
    # Print stats for paku
    print(f'Species: {paku}')
    print(f'Attack: {paku_stats[0]}')
    print(f'Defense: {paku_stats[1]}')
    print(f'Speed: {paku_stats[2]}')


def main():
    # Welcome message and dex initialization
    print('Welcome to Pakudex: Tracker Extraordinaire!')

    # Get valid max size
    while True:
        try:
            max_capacity = int(input('Enter max capacity of the Pakudex: '))
            if max_capacity < 0:
                raise ValueError
            break
        except ValueError:
            print('Please enter a valid size.')

    dex = pakudex(max_capacity)
    print(f'The Pakudex can hold {max_capacity} species of Pakuri.')

    # Run main loop
    while True:
        selection = menu()

        if selection == 1:
            # Print list of all current Pakuris
            print_pakus(dex.get_species_array())

        elif selection == 2:
            # Print stats for specific paku
            paku = input('Enter the name of the species to display: ')
            stats = dex.get_stats(paku)
            
            # If in dex
            if stats:
                show_paku(paku, stats)

            else:
                print('Error: No such Pakuri!')

        elif selection == 3:
            # Add new paku
            if dex.is_full:
                print('Error: Pakudex is full!')
                continue

            species = input('Enter the name of the species to add: ')   
            # Check unique         
            if dex.add_pakuri(species):
                print(f'Pakuri species {species} successfully added!')

            else:
                print('Error: Pakudex already contains this species!')

        elif selection == 4:
            # Evolve selected Paku
            species = input('Enter the name of the species to evolve: ')
            if dex.evolve_species(species):
                print(f'{species} has evolved!')
            else:
                print('Error: No such Pakuri!')

        elif selection == 5:
            dex.sort_pakuri()
            print('Pakuri have been sorted!')

        elif selection == 6:
            # Exit loop to quit
            print('Thanks for using Pakudex! Bye!')
            break


if __name__ == "__main__":
    main()
    