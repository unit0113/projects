#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>

#define CARDS 52
#define FACES 13
#define SUITS 4
#define SHUFFLES 3
#define HAND_SIZE 5

typedef struct {
    size_t face;
    size_t suit;
} Card;

std::vector<Card> initialize_deck();
void shuffle_deck(std::vector<Card> &deck);
Card deal_card(std::vector<Card> &deck);
std::vector<Card> deal_hand(std::vector<Card> &deck);
void print_deck(std::vector<Card> &deck);
void print_card(Card &card);

int main() {

    std::vector<Card> deck = initialize_deck();
    shuffle_deck(deck);
    
    std::vector<Card> hand = deal_hand(deck);
    print_deck(hand);
    




    return 0;
}


std::vector<Card> initialize_deck() {
    std::vector<Card> deck;
    for (size_t i = 0; i < CARDS; i++) {
        deck.push_back(Card {i % FACES, i % SUITS});
    }

    return deck;

}


void shuffle_deck(std::vector<Card> &deck) {
    auto rng = std::default_random_engine();
    for (size_t i = 0; i < SHUFFLES; i++) {
        std::shuffle(std::begin(deck), std::end(deck), rng);
    }
}


void print_deck(std::vector<Card> &deck) {
    for (Card card: deck) {
        print_card(card);
    }
}


void print_card(Card &card) {
    static const std::vector<std::string> faces = {"Ace", "Deuce", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"};

    static const std::vector<std::string> suits = {"Hearts", "Diamonds", "Clubs", "Spades"};

    auto face_width = std::setw(5);

    std::cout << face_width << std::left << faces[card.face] << " of " << suits[card.suit] << std::endl;
}


Card deal_card(std::vector<Card> &deck) {
    srand(time(NULL));
    size_t rand_num = rand() % deck.size();
    Card picked_card = deck[rand_num];
    deck.erase(deck.begin() + rand_num);

    return picked_card;
}


std::vector<Card> deal_hand(std::vector<Card> &deck) {
    std::vector<Card> hand;
    for (size_t i = 0 ; i < HAND_SIZE; i++) {
        hand.push_back(deal_card(deck));
    }
    return hand;
}