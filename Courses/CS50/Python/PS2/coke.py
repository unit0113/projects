coins = [25, 10, 5]
TOTAL_NEEDED = 50

total = 0
while total < 50:
    print(f"Amount Due: {TOTAL_NEEDED - total}")
    coin_in = int(input("Insert Coin: "))
    if coin_in in coins:
        total += coin_in

print(f"Change Owed: {total - TOTAL_NEEDED}")