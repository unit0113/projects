from numb3rs import validate


def test_good():
    assert validate("127.0.0.1") == True

def test_high():
    assert validate("255.255.255.255") == True

def test_low():
    assert validate("0.0.0.0") == True

def test_long():
    assert validate("1.2.3.1000") == False

def test_str():
    assert validate("cat") == False

def test_bad_middle():
    assert validate("127.456.76.44") == False
