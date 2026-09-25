from backend.zones import Zone


def test_normalized_zone_scales_to_frame_pixels():
    z = Zone(id="a", name="A", polygon=[(0.1, 0.2), (0.5, 0.2), (0.5, 1.0)], allowed_direction=(1.0, 0.0))
    px = z.to_pixels(640, 480)
    assert px.polygon == [(64.0, 96.0), (320.0, 96.0), (320.0, 480.0)]
    assert px.allowed_direction == (640.0, 0.0)
    assert z.polygon[0] == (0.1, 0.2), "original is not mutated"


def test_pixel_zone_from_old_clients_is_kept_and_normalizable():
    z = Zone(id="a", name="A", polygon=[(64, 96), (320, 96), (320, 480)])
    assert not z.is_normalized
    assert z.to_pixels(640, 480) is z
    n = z.to_normalized(640, 480)
    assert n.is_normalized and n.polygon[0] == (0.1, 0.2)


def test_pixel_zone_without_frame_size_stays_pixels():
    z = Zone(id="a", name="A", polygon=[(64, 96), (320, 96), (320, 480)])
    assert z.to_normalized(0, 0) is z
