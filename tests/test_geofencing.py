"""Unit tests for the geofencing / compliance layer. No GPU, no model — pure logic."""
from ppe_studio.geofencing import (
    Detection,
    Severity,
    Zone,
    evaluate,
    point_in_polygon,
)

SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_point_inside_square():
    assert point_in_polygon((0.5, 0.5), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((1.5, 0.5), SQUARE) is False
    assert point_in_polygon((-0.1, 0.5), SQUARE) is False


def test_degenerate_polygon():
    assert point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)]) is False


def test_concave_polygon():
    # L-shape: (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
    l_shape = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    assert point_in_polygon((0.5, 1.5), l_shape) is True   # in the tall arm
    assert point_in_polygon((1.5, 1.5), l_shape) is False  # in the notch


def _det(role, box, cls="head", conf=0.9):
    return Detection(cls_name=cls, role=role, conf=conf, xyxy=box)


def test_no_helmet_inside_critical_zone_is_critical():
    zone = Zone("press", tuple(SQUARE), frozenset({"helmet"}), Severity.CRITICAL)
    # bare head whose foot point (bottom-center) lands inside the zone
    head = _det("ppe_violation", (0.4, 0.3, 0.6, 0.8))
    violations = evaluate([head], [zone])
    assert any(v.rule == "no_helmet" and v.severity == Severity.CRITICAL for v in violations)


def test_no_helmet_off_zone_is_site_warning():
    zone = Zone("press", ((0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)), severity=Severity.CRITICAL)
    head = _det("ppe_violation", (0.7, 0.7, 0.9, 0.95))  # outside the small zone
    violations = evaluate([head], [zone])
    assert len(violations) == 1
    assert violations[0].severity == Severity.WARNING


def test_helmet_present_no_violation():
    zone = Zone("press", tuple(SQUARE), frozenset({"helmet"}), Severity.CRITICAL)
    helmet = _det("ppe_ok", (0.4, 0.3, 0.6, 0.8), cls="helmet")
    assert evaluate([helmet], [zone]) == []


def test_person_in_zone_without_helmet_is_intrusion():
    zone = Zone("press", tuple(SQUARE), frozenset({"helmet"}), Severity.CRITICAL)
    person = _det("subject", (0.4, 0.2, 0.6, 0.9), cls="person")
    violations = evaluate([person], [zone])
    assert any(v.rule == "intrusion" and v.severity == Severity.CRITICAL for v in violations)


def test_person_with_overlapping_helmet_no_intrusion():
    zone = Zone("press", tuple(SQUARE), frozenset({"helmet"}), Severity.CRITICAL)
    person = _det("subject", (0.4, 0.2, 0.6, 0.9), cls="person")
    helmet = _det("ppe_ok", (0.42, 0.2, 0.58, 0.35), cls="helmet")
    violations = evaluate([person, helmet], [zone])
    assert not any(v.rule == "intrusion" for v in violations)
