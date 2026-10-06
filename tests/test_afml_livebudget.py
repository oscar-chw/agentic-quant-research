import pytest

from pmlab.afml.livebudget import CEILING_MS, DEADLINE_MS, SecondCost, WindowBudget


def _slow(second, events=5):
    return SecondCost(second, feed_ms=CEILING_MS, decide_ms=1.0, record_ms=1.0, events=events)


@pytest.mark.case
def test_idle_seconds_stay_out_of_the_alert_evidence():
    """SecondCost.events is documented as keeping idle seconds out of the evidence so an idle stretch cannot dilute a
    slow one. It was never read: 3 slow seconds among 290 idle ones reported 'of the window's 293 seconds' and divided
    the mean_*_ms by 293."""
    b = WindowBudget(over=3)
    for t in range(0, 290):
        assert b.on_second(SecondCost(t, 0.0, 0.0, 0.0, events=0)) is None
    alerts = [b.on_second(_slow(t)) for t in (290, 291, 292)]
    alert = alerts[-1]
    assert alert is not None and alert["evidence"]["rule"] == "ceiling"
    assert alert["evidence"]["seconds_seen"] == 3 and alert["evidence"]["seconds_over"] == 3
    assert alert["evidence"]["mean_feed_ms"] == pytest.approx(CEILING_MS)         # not CEILING_MS * 3 / 293
    assert "3 of the window's 3 seconds" in alert["text"]
    assert b.report()["seconds"] == 293                                          # running totals still count every second


@pytest.mark.case
def test_idle_seconds_do_not_count_toward_the_over_ceiling_rule_and_a_busy_one_does():
    b = WindowBudget(over=2)
    assert b.on_second(_slow(0, events=0)) is None
    assert b.on_second(_slow(1, events=0)) is None                               # idle: costs nothing, proves nothing
    assert b.on_second(_slow(2)) is None
    assert b.on_second(_slow(3)) is not None                                     # the second busy one reaches over=2


@pytest.mark.case
def test_an_idle_second_that_misses_the_deadline_still_alerts():
    b = WindowBudget()
    alert = b.on_second(SecondCost(7, DEADLINE_MS + 1, 0.0, 0.0, events=0))
    assert alert is not None and alert["evidence"]["rule"] == "deadline"
