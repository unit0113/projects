import requests
import sys

def main():
    if len(sys.argv) != 2:
        sys.exit("Missing command-line argument")
    try:
        amount = float(sys.argv[1])
    except ValueError:
        sys.exit("Command-line argument is not a number")

    try:
        response = requests.get(r"https://api.coindesk.com/v1/bpi/currentprice.json")
    except requests.RequestException:
        sys.exit("Network Error")

    price = response.json()['bpi']['USD']['rate_float']
    total = float(price) * amount
    print(f'${total:,.4f}')



if __name__ == "__main__":
    main()
