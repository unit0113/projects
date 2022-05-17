import random
import time


# Generate Deck
CARDS = []
for num in range(1, 14):
    CARDS += [num] * 4

# Initialize card values
CARD_VALUES = [1, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 11]


def hit(deck):
    card = deck.pop(random.randrange(len(deck)))
    return card, deck


def deal(deck):
    hand = []
    for _ in range(2):
        card, deck = hit(deck)
        hand.append(card)

    return hand, deck


def draw_card(hand, hand_value, deck):
    card, deck = hit(deck)
    print(f'The card is a {card_name(card)}')
    hand_value += CARD_VALUES[card]
    hand.append(card)

    if hand_value > 21 and 1 in hand:
        hand_value -= 10
        hand.remove(1)
        hand.append(0)

    return hand, hand_value, deck


def play_AI(deck, hand):
    print(f'The dealer reveals a {card_name(hand[1])}')
    hand_value = get_hand_value(hand)
    print(f'The dealer\'s hand is currently {hand_value}')
    while hand_value < 17:
        time.sleep(1)
        print('Dealer hits')
        hand, hand_value, deck = draw_card(hand, hand_value, deck)
        print(f'The dealer now has {hand_value}')
        print('*' * 20)

    if hand_value <= 21:
        print(f'Dealer stays at {hand_value}')

    return hand_value


def get_hand_value(hand):
    return sum([CARD_VALUES[card] for card in hand])


def card_name(card):
    if card == 1:
        return 'Ace'
    elif card == 11:
        return 'Jack'
    elif card == 12:
        return 'Queen'
    elif card == 13:
        return 'King'
    return card


def play_human(deck, ai_hand):
    hand, deck = deal(deck)
    hand_value = get_hand_value(hand)
    ai_card = CARD_VALUES[ai_hand]
    while hand_value <= 21:
        print(f'Current hand value is {hand_value}')
        print(f'The dealer currently shows {ai_card}')
        print('Press 1 to hit')
        print('Press 2 to stay')
        selection = input('Make selection: ')
        print('*' * 20)

        if selection == '2':
            print(f'Player stays on {hand_value}')
            return hand_value, deck

        hand, hand_value, deck = draw_card(hand, hand_value, deck)

    return hand_value, deck


def check_winner(human_hand_value, ai_hand_value):
    print('*' * 20)
    print('*' * 20)
    print(f'The player finished with {human_hand_value}')
    print(f'The dealer finished with {ai_hand_value}')
    if human_hand_value == ai_hand_value:
        print('Draw!')
    elif human_hand_value > ai_hand_value:
        print('The player wins!')
    else:
        print('The dealer wins!')

    selection = input('Press 1 to play again, or anything else to exit: ')
    if selection == '1':
        print('*' * 20)
        print('*' * 20)
        main()


def main():
    random.shuffle(CARDS)
    ai_hand, deck = deal(CARDS)
    human_hand_value, deck = play_human(deck, ai_hand[0])
    if human_hand_value > 21:
        print('Bust! Dealer wins!')
        return

    ai_hand_value = play_AI(deck, ai_hand)
    if ai_hand_value > 21:
        print('Bust! Player wins!')
        return

    check_winner(human_hand_value, ai_hand_value)


if __name__ == "__main__":
    main()