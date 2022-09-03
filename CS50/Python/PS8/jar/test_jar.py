import pytest
from jar import Jar


def test_default_construct():
    jar = Jar()
    assert jar.capacity == 12

def test_construct():
    jar = Jar(13)
    assert jar.capacity == 13

def test_neg_construct():
    with pytest.raises(ValueError):
        jar = Jar(-13)

def test_bad_type_construct():
    with pytest.raises(ValueError):
        jar = Jar("COOKIE!!")

def test_add_cookie():
    jar = Jar()
    jar.deposit(5)
    assert jar.size == 5
    jar.withdraw(3)
    assert jar.size == 2
    with pytest.raises(ValueError):
        jar.deposit(20)

def test_print_cookie():
    jar = Jar()
    jar.deposit(3)
    assert str(jar) == "🍪🍪🍪"