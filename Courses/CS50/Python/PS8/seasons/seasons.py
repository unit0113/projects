from datetime import date
import inflect
import sys


def main():
    try:
        year, month, day = input("Enter Date of birth: ").split('-')
        date_birth = date(int(year), int(month), int(day))
    except:
        sys.exit(1)

    delta_minutes = calc_delta_t(date_birth)
    wordy_output(delta_minutes)


def calc_delta_t(date_birth):
    date_now = date.today()
    delta_t = date_now - date_birth
    return delta_t.days * 24 * 60


def wordy_output(minutes):
    p = inflect.engine()
    print(p.number_to_words(minutes, andword="").capitalize() + " minutes")


if __name__ == "__main__":
    main()