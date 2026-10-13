import pytest

from jpegoverdrive.scheduler import AdaptiveLPIPSScheduler


def test_scheduler_switches_to_vgg() -> None:
    """Use VGG when its normalized distortion dominates."""
    scheduler = AdaptiveLPIPSScheduler()

    scheduler.update(alex=0.90, vgg=1.00)

    assert scheduler.select(1) == "both"
    assert scheduler.select(2) == "vgg"
    assert scheduler.select(10) == "both"


def test_scheduler_near_tie() -> None:
    """Keep both networks active when distortions are within the threshold."""
    scheduler = AdaptiveLPIPSScheduler(threshold=0.02)

    scheduler.update(alex=0.99, vgg=1.00)

    assert scheduler.select(2) == "both"

    scheduler.update(alex=0.95, vgg=1.00)

    assert scheduler.select(3) == "vgg"


def test_scheduler_switches_to_alex() -> None:
    """Use AlexNet when its normalized distortion dominates."""
    scheduler = AdaptiveLPIPSScheduler()

    scheduler.update(alex=1.00, vgg=0.90)

    assert scheduler.select(1) == "both"
    assert scheduler.select(2) == "alex"
    assert scheduler.select(10) == "both"


def test_scheduler_returns_to_joint() -> None:
    """Resume joint evaluation when normalized distortions converge."""
    scheduler = AdaptiveLPIPSScheduler()

    scheduler.update(alex=0.90, vgg=1.00)
    assert scheduler.select(2) == "vgg"

    scheduler.update(alex=0.99, vgg=1.00)
    assert scheduler.select(3) == "both"


@pytest.mark.parametrize("check_every", [0, -1])
def test_invalid_check_interval(check_every: int) -> None:
    """Reject invalid evaluation intervals."""
    with pytest.raises(ValueError):
        AdaptiveLPIPSScheduler(check_every=check_every)
