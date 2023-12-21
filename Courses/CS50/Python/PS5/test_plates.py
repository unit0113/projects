from plates import is_valid


def test_alpha():
    assert is_valid("HELLO") == True

def test_alpha_numeric():
    assert is_valid("CS50") == True

def test_lead_numeric():
    assert is_valid("A2") == False

def test_lead_0():
    assert is_valid("CS05") == False

def test_short():
    assert is_valid("H") == False

def test_long():
    assert is_valid("TOODAMNLONG") == False

def test_middle_numeric():
    assert is_valid("CS50P") == False

def test_punc():
    assert is_valid("PI3.14") == False
