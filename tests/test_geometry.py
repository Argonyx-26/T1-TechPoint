from backend.geometry import angle_between_deg, bbox_centroid, euclidean, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_polygon():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside_polygon():
    assert point_in_polygon((15, 5), SQUARE) is False


def test_bbox_centroid():
    assert bbox_centroid((0, 0, 10, 20)) == (5.0, 10.0)


def test_euclidean():
    assert euclidean((0, 0), (3, 4)) == 5.0


def test_angle_between_deg_opposite_vectors():
    assert abs(angle_between_deg((1, 0), (-1, 0)) - 180.0) < 1e-6


def test_angle_between_deg_same_vector():
    assert angle_between_deg((1, 0), (1, 0)) < 1e-6
