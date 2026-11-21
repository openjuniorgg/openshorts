from camera_inset import (
    INSET_ASPECT,
    inset_box,
    inset_filtergraph,
    is_cornered,
    nearest_corner,
    usable,
)

HD = (1920, 1080)


class TestNearestCorner:
    def test_reads_the_quadrant(self):
        assert nearest_corner([10, 10, 100, 100], *HD) == ("left", "top")
        assert nearest_corner([1800, 980, 100, 100], *HD) == ("right", "bottom")
        assert nearest_corner([1800, 10, 100, 100], *HD) == ("right", "top")
        assert nearest_corner([10, 980, 100, 100], *HD) == ("left", "bottom")


class TestIsCornered:
    def test_small_subject_against_an_edge_qualifies(self):
        # A webcam pinned to the bottom-right: the exact case YOLO reports as a
        # short, wide upper-body box.
        assert is_cornered([1500, 900, 350, 120], *HD)

    def test_a_presenter_filling_the_shot_is_rejected(self):
        assert not is_cornered([600, 100, 700, 900], *HD)

    def test_a_small_but_centred_subject_is_rejected(self):
        assert not is_cornered([900, 500, 150, 100], *HD)

    def test_box_not_touching_the_bottom_still_counts(self):
        # Measured failure: an inset pinned to the right edge sat 18% clear of
        # the bottom because the detector stopped at the chest. Requiring two
        # edges threw it away.
        assert is_cornered([991 * 1.5, 502 * 1.5, 246 * 1.5, 86 * 1.5], *HD)

    def test_a_tall_subject_is_never_an_inset(self):
        assert not is_cornered([1700, 100, 200, 800], *HD)


class TestInsetBox:
    def test_box_stays_inside_the_frame(self):
        for box in ([0, 980, 200, 100], [1800, 0, 120, 120],
                    [1850, 1000, 70, 80]):
            x, y, w, h = inset_box(box, *HD)
            assert 0 <= x and x + w <= 1920
            assert 0 <= y and y + h <= 1080

    def test_bottom_anchored_inset_reaches_the_bottom_edge(self):
        # The head sits above the detection, so the box must extend upward
        # while staying pinned to the edge the inset actually touches.
        x, y, w, h = inset_box([1500, 950, 300, 100], *HD)
        assert y + h == 1080
        assert y < 950

    def test_height_follows_the_16_9_assumption(self):
        # A wide, flat torso detection must still yield a webcam-shaped box.
        _x, _y, w, h = inset_box([1000, 800, 300, 80], *HD)
        assert h >= w / INSET_ASPECT - 1

    def test_box_grows_around_the_subject(self):
        x, y, w, h = inset_box([1600, 900, 200, 120], *HD)
        assert w >= 200 and h >= 120


class TestFaceDetections:
    """A face box grown like a torso gave a box that started below the
    forehead (Excel walkthrough, 17-oct-2026)."""

    def test_a_face_in_the_corner_keeps_the_whole_head(self):
        face = [1054, 497, 128, 128]          # measured, 1280x720
        x, y, w, h = inset_box(face, 1280, 720, face=True)
        assert y <= 497 - 0.3 * 128
        assert y + h == 720
        assert w >= 3 * 128 - 1

    def test_a_face_box_gives_a_webcam_shaped_box(self):
        _x, _y, w, h = inset_box([1088, 458, 96, 96], 1280, 720, face=True)
        assert abs(w / h - INSET_ASPECT) < 0.05

    def test_face_and_body_detections_agree(self):
        # Same webcam, one sample found the face and the next only the body:
        # the boxes must land on the same inset for stable_box to agree.
        by_face = inset_box([77, 845, 125, 125], 1920, 1080, face=True)
        by_body = inset_box([0, 838, 231, 96], 1920, 1080)
        cx = lambda b: b[0] + b[2] / 2.0
        assert abs(cx(by_face) - cx(by_body)) <= 1920 * 0.05


class TestUsable:
    def test_rejects_a_speck(self):
        assert not usable((0, 0, 40, 20), 1080)

    def test_rejects_something_that_fills_the_frame(self):
        assert not usable((0, 0, 1920, 900), 1080)

    def test_accepts_a_normal_webcam(self):
        assert usable((1500, 850, 350, 200), 1080)


class TestInsetFiltergraph:
    def test_graph_places_screen_and_camera(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        assert "split=3" in graph
        assert "[screen]" in graph and "[cam]" in graph
        assert graph.endswith("[v]")

    def test_webcam_on_top_by_default(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        # Streamer webcam is placed on top, gameplay/screen is placed on bottom
        assert "[bg][cam]overlay=x=" in graph
        assert "[withcam];" in graph
        assert "[withcam][screen]overlay=x=0:y=576" in graph
        assert graph.endswith(",setsar=1[v]")

    def test_webcam_on_top_explicit_true(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200), webcam_on_top=True)
        assert "[bg][cam]overlay=x=" in graph
        assert "[withcam][screen]overlay=x=0:y=576" in graph

    def test_webcam_on_top_false_places_screen_on_top(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200), webcam_on_top=False)
        # Previous behavior: screen on top, camera below
        assert "[bg][screen]overlay=x=0:y=" in graph
        assert "[withscreen];" in graph
        assert "[withscreen][cam]overlay=x=" in graph
        assert graph.endswith(",setsar=1[v]")

    def test_webcam_on_top_environment_variable(self, monkeypatch):
        monkeypatch.setenv("WEBCAM_ON_TOP", "0")
        graph_bottom = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        assert "[bg][screen]overlay=x=0:y=" in graph_bottom
        assert "[withscreen][cam]overlay=x=" in graph_bottom

        monkeypatch.setenv("WEBCAM_ON_TOP", "true")
        graph_top = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        assert "[bg][cam]overlay=x=" in graph_top
        assert "[withcam][screen]overlay=x=0:y=576" in graph_top

    def test_gameplay_fills_the_bottom_three_quarters(self):
        # The stream layout: webcam top 30%, gameplay bottom 70% (1344 of 1920).
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        screen = graph.split("[sa]")[-1].split("[screen]")[0]
        assert "scale=1080:1344" in screen
        # Centred crop of the 16:9 source to 1080:1344: 868 px wide, full height.
        assert "crop=w=868:h=1080:x=526:y=0" in screen

    def test_gameplay_keeps_its_aspect_when_cropped(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        screen = graph.split("[sa]")[-1].split("[screen]")[0]
        w = int(screen.split("crop=w=")[1].split(":")[0])
        h = int(screen.split("h=")[1].split(":")[0])
        assert abs((w / h) - (1080 / 1344)) < 0.01

    def test_webcam_fills_the_band_with_the_camera_alone(self):
        # The camera box is trimmed to the band's shape and never grown past it:
        # 350x200 becomes the 350x186 crop inside it, filling the 1080x576 band.
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        assert "crop=w=350:h=186:x=1500:y=882" in graph
        assert "scale=1080:576[cam]" in graph
        assert "[bg][cam]overlay=x=0:y=0" in graph

    def test_the_camera_is_never_stretched(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 880, 350, 200))
        crop = graph.split("[ca]")[-1].split("[cam]")[0]
        cw, ch = int(crop.split("w=")[1].split(":")[0]), int(crop.split("h=")[1].split(":")[0])
        assert abs((cw / ch) - (1080 / 576)) < 0.01

    def test_a_square_webcam_is_trimmed_to_the_band_inside_its_box(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1400, 500, 300, 300))
        crop = graph.split("[ca]")[-1].split("[cam]")[0]
        cx = int(crop.split("x=")[1].split(":")[0])
        cy = int(crop.split("y=")[1].split(",")[0])
        cw, ch = int(crop.split("w=")[1].split(":")[0]), int(crop.split("h=")[1].split(":")[0])
        assert abs((cw / ch) - (1080 / 576)) < 0.01
        # the crop lies inside the webcam box (x 1400..1700, y 500..800): no gameplay in the band
        assert cx >= 1400 and cx + cw <= 1700 and cy >= 500 and cy + ch <= 800
        assert "[bg][cam]overlay=x=0:y=0" in graph

    def test_webcam_band_is_30_percent_of_the_frame(self):
        import re
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1500, 700, 300, 290))
        assert "[withcam][screen]overlay=x=0:y=576" in graph  # gameplay starts at 30%
        fitted_h = int(re.search(r"scale=(\d+):(\d+)\[cam\]", graph).group(2))
        assert fitted_h <= 576

    def test_crop_coordinates_stay_inside_the_source(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1850, 1000, 70, 80))
        cam = graph.split("[ca]")[-1].split("[cam]")[0]
        x = int(cam.split("x=")[1].split(":")[0])
        w = int(cam.split("w=")[1].split(":")[0])
        assert x >= 0 and x + w <= 1920

    def test_every_dimension_is_even(self):
        graph = inset_filtergraph(1920, 1080, 1080, 1920, (1501, 881, 351, 201))
        cam = graph.split("[ca]")[-1].split("[cam]")[0]
        for key in ("w=", "h=", "x=", "y="):
            assert int(cam.split(key)[1].split(":")[0].split(",")[0]) % 2 == 0
