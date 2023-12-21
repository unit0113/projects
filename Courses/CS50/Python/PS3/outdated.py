months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"]

while True:
    raw_date = input("Date: ").strip()
    if '/' in raw_date:
        month, day, year = raw_date.split('/')

    else:
        month, day, year = raw_date.split(' ')
        if ',' not in day:
            continue

        day = day.replace(',', '')
        try:
            month = months.index(month) + 1
        except ValueError:
            continue

    try:
        day = int(day)
        month = int(month)
    except ValueError:
        continue

    if not 0 < day <= 31 or not 0 < month <= 12:
        continue

    break

print(f'{year}-{month:02}-{day:02}')