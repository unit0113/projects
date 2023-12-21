import pytest
from seasons import calc_delta_t
from datetime import date, timedelta


def test_long():
    date_now = date.today()
    test_date = date_now - timedelta(days = 365)
    assert calc_delta_t(test_date) == 525600