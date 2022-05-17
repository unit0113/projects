import p1_random as p1
RNG = p1.P1Random()


class Game:
    def __init__(self):
        self.game_number = 0
        self.player_wins = 0
        self.dealer_wins = 0


    def menu(self):
        """Display menu and get input from user. Repeat menu if invalid choice

        Returns:
            int: chosen menu option
        """

        response = 0

        # Loop through menu until valid response
        while response < 1 or response > 4:
            print('1. Get another card')
            print('2. Hold hand')
            print('3. Print statistics')
            print('4. Exit')

            print()
            response = int(input('Choose an option: '))
            print()

            if response < 1 or response > 4:
                print('Invalid input!')
                print('Please enter an integer value between 1 and 4.')
                print()

        return response


    def print_statistics(self):
        """Prints game statistics

        """

        print(f'Number of Player wins: {self.player_wins}')
        print(f'Number of Dealer wins: {self.dealer_wins}')
        print(f'Number of tie games: {self.game_number - 1 - self.player_wins - self.dealer_wins}')
        print(f'Total # of games played is: {self.game_number - 1}')
        print(f'Percentage of Player wins: {(100 * self.player_wins / (self.game_number - 1)):.1f}%')
        print()


    def draw_card(self):
        """Draw a card, print the result, and return the value of the card

        Returns:
            int: hand value of the drawn card
        """
        card_names = [None, 'ACE'] + [str(num) for num in range(2, 11)] + ['Jack', 'Queen', 'King']
        card_values = [None] + [num for num in range(1, 11)] + [10] * 3

        # Retrieve and print card with proper name
        card = RNG.next_int(13) + 1
        print(f'Your card is a {card_names[card]}!')

        return card_values[card]


    def draw_card_AI(self):
        """Draw a card and return the value of the card. AI version does not print anything

        Returns:
            int: hand value of the drawn card
        """
        card_values = [None] + [num for num in range(1, 11)] + [10] * 3

        # Retrieve the card
        card = RNG.next_int(13) + 1

        return card_values[card]


    def play_dealer(self):
        """Play the dealers hand. Must hit when less than 17

        Returns:
            int: Dealer hand value
        """
        hand = 0
        while hand < 17:
            hand += self.draw_card_AI()

        return hand


    def print_hand(self, hand):
        """Prints hand value

        Args:
            hand (int): Current value of hand
        """
        print(f'Your hand is: {hand}')
        print()


    def check_early_end(self, hand):
        """Check if player busted or got a blackjack

        Args:
            hand (int): Current value of hand

        Returns:
            bool: True if game ends early
        """

        # Check for bust
        if hand > 21:
            print('You exceeded 21! You lose.')
            self.dealer_wins += 1

        # Check for blackjack
        elif hand == 21:
            print('BLACKJACK! You win!')
            self.player_wins += 1

        else:
            return

        # Cleanup
        print()
        self.play()


    def endgame(self, player_hand, dealer_hand):
        """Determines winner and prints result

        Args:
            player_hand (int): Value of player's hand
            dealer_hand (int): Value of dealer's hand
        """

        # Print hands
        print(f'Dealer\'s hand: {dealer_hand}')
        self.print_hand(player_hand)

        # Determine winner
        if dealer_hand > 21 or player_hand > dealer_hand:
            print('You win!')
            self.player_wins += 1
        elif dealer_hand == player_hand:
            print('It\'s a tie! No one wins!')
        else:
            print('Dealer wins!')      
            self.dealer_wins += 1  

        print()


    def play(self):
        """Main function. Prints game intro, loops through human turns, plays AI, and determines winner.
        """

        # Game intro
        self.game_number += 1
        print(f'START GAME #{self.game_number}')
        print()

        # Initial card draw
        hand = self.draw_card()

        # Player turn
        while True:
            self.print_hand(hand)
            response = self.menu()
            if response == 1:
                hand += self.draw_card()
            elif response == 2:
                break
            elif response == 3:
                self.print_statistics()
            elif response == 4:
                quit()

            self.check_early_end(hand)

        # Play dealer
        dealer_hand = self.play_dealer()

        # Determine winner
        self.endgame(hand, dealer_hand)

        # Start next game
        self.play()


def main():
    game = Game()
    game.play()


if __name__ == "__main__":
    main()