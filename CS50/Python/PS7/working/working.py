import re
import sys


def main():
    print(convert(input("Hours: ")))


def convert(s):
    re_style_long = re.compile(r"(\d\d?:\d\d) (AM|PM) to (\d\d?:\d\d) (AM|PM)")
    re_style_short = re.compile(r"(\d\d?) (AM|PM) to (\d\d?) (AM|PM)")

    if match := re.search(re_style_long, s):
        return (convert_long(match))
    elif match := re.search(re_style_short, s):
        return (convert_short(match))
    else:
        raise ValueError


def convert_long(match):
    time_1, time_type_1, time_2, time_type_2 = match.groups()
    time_1_hr, time_1_min = time_1.split(':')
    time_2_hr, time_2_min = time_2.split(':')
    time_1_hr, time_2_hr = int(time_1_hr), int(time_2_hr)

    check_minutes(time_1_min)
    check_minutes(time_2_min)

    time_1_hr = time_convert(time_1_hr, time_type_1)
    time_2_hr = time_convert(time_2_hr, time_type_2)

    return f"{time_1_hr:02}:{time_1_min} to {time_2_hr:02}:{time_2_min}"


def check_minutes(min):
    if 0 <= int(min) <= 59:
        return
    else:
        raise ValueError


def convert_short(match):
    time_1_hr, time_type_1, time_2_hr, time_type_2 = match.groups()
    time_1_hr, time_2_hr = int(time_1_hr), int(time_2_hr)

    time_1_hr = time_convert(time_1_hr, time_type_1)
    time_2_hr = time_convert(time_2_hr, time_type_2)

    return f"{time_1_hr:02}:00 to {time_2_hr:02}:00"


def time_convert(time_hr, time_type):
    if time_hr == 12:
        if time_type == "AM":
            return 0
        if time_type == "PM":
            return 12
    elif time_type == "PM":
        return (time_hr + 12) % 24
    else:
        return time_hr


if __name__ == "__main__":
    main()