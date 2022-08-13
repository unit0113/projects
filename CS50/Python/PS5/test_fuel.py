import pytest
from fuel import convert, gauge


def test_convert_correct():
    assert convert("3/4") == 75

def test_convert_improper():
    with pytest.raises(ValueError):
        convert("6/4")

def test_convert_non_numeric():
    with pytest.raises(ValueError):
        convert("a/4")

def test_convert_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        convert("3/0")

def test_gauge_f():
    assert gauge(99) == "F"

def test_gauge_e():
    assert gauge(1) == "E"

def test_gauge_nominal():
    assert gauge(50) == "50%"
