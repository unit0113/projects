import pytest
from working import convert


def test_long():
    assert convert("9:00 AM to 5:00 PM") == "09:00 to 17:00"

def test_short():
    assert convert("9 AM to 5 PM") == "09:00 to 17:00"

def test_invert():
    assert convert("10 PM to 8 AM") == "22:00 to 08:00"

def test_minute():
    assert convert("10:30 PM to 8:50 AM") == "22:30 to 08:50"

def test_long_midnight():
    assert convert("12:00 AM to 12:00 PM") == "00:00 to 12:00"

def test_short_midnight():
    assert convert("12 AM to 12 PM") == "00:00 to 12:00"

def test_big_minute():
    with pytest.raises(ValueError):
        convert("9:60 AM to 5:60 PM")

def test_no_to_short():
    with pytest.raises(ValueError):
        convert("9 AM - 5 PM")

def test_no_to_long():
    with pytest.raises(ValueError):
        convert("09:00 AM - 17:00 PM")

def test_no_meridian():
    with pytest.raises(ValueError):
        convert("9:72 to 6:30")