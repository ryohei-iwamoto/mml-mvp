import os
import tempfile
import unittest

import cv2
import ezdxf
import numpy as np

from mml.draw import draw_dxf
from mml.pipeline import run_pipeline
from mml.utils import read_json


def _blank_canvas(width, height):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw_plate(img, rect, holes=None, bend=None):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    for (cx, cy, r) in holes or []:
        cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
    if bend:
        (x1, y1), (x2, y2) = bend
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return img


class PipelineTests(unittest.TestCase):
    def _write_image(self, path, img):
        cv2.imwrite(path, img)

    def test_case_1_simple_plate(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _blank_canvas(240, 140)
            holes = [(80, 60, 8), (160, 60, 8), (80, 100, 8), (160, 100, 8)]
            _draw_plate(img, (20, 20, 220, 120), holes=holes)
            image_path = os.path.join(tmp, "case1.png")
            self._write_image(image_path, img)

            out_dir = os.path.join(tmp, "out1")
            outputs = run_pipeline(
                image_path,
                out_dir,
                params={"plate_width_mm": 100, "hole_standard": "M5", "thickness_mm": 2.0},
            )

            mml = read_json(outputs["mml"])
            self.assertEqual(mml["part"], "Bracket")
            self.assertEqual(len(mml["geometry"]["holes"]), 4)
            self.assertTrue(os.path.exists(outputs["dxf"]))

    def test_case_2_plate_with_bend(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _blank_canvas(240, 140)
            bend = ((120, 30), (120, 110))
            _draw_plate(img, (20, 20, 220, 120), bend=bend)
            image_path = os.path.join(tmp, "case2.png")
            self._write_image(image_path, img)

            out_dir = os.path.join(tmp, "out2")
            outputs = run_pipeline(
                image_path,
                out_dir,
                params={"plate_width_mm": 100, "bend_angle_deg": 90, "bend_radius_mm": 2.0},
            )

            mml = read_json(outputs["mml"])
            self.assertIn("bend", mml["geometry"])
            doc = ezdxf.readfile(outputs["dxf"])
            layers = {layer.dxf.name for layer in doc.layers}
            self.assertIn("BEND", layers)

    def test_case_3_ambiguous_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = _blank_canvas(240, 140)
            holes = [(80, 60, 6), (160, 60, 10)]
            _draw_plate(img, (20, 20, 220, 120), holes=holes)
            image_path = os.path.join(tmp, "case3.png")
            self._write_image(image_path, img)

            out_dir = os.path.join(tmp, "out3")
            outputs = run_pipeline(
                image_path,
                out_dir,
                params={"plate_width_mm": 100, "unify_holes": True},
            )

            report = read_json(outputs["report"])
            self.assertIn("hole_size_normalized", report.get("decisions", []))

    def test_draw_dxf_uses_three_view_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "drawing.dxf")
            mml = {
                "part": "Bracket",
                "units": "mm",
                "geometry": {
                    "outline": {
                        "type": "polygon",
                        "points_mm": [[0, 0], [100, 0], [100, 60], [0, 60]],
                    },
                    "holes": [
                        {"center_mm": [25, 30], "diameter_mm": 10.0, "standard": "M8"},
                        {"center_mm": [75, 30], "diameter_mm": 10.0, "standard": "M8"},
                    ],
                    "bend": {"line_mm": [[50, 0], [50, 60]], "angle_deg": 90, "inner_radius_mm": 2.0},
                },
                "constraints": [{"kind": "min_thickness", "value_mm": 4.0}],
            }
            draw_dxf(mml, out_path)

            doc = ezdxf.readfile(out_path)
            layers = {layer.dxf.name for layer in doc.layers}
            self.assertTrue({"OUTLINE", "HOLES", "BEND", "CENTER", "HIDDEN", "TEXT"}.issubset(layers))
            texts = list(doc.modelspace().query("TEXT"))
            joined = "\n".join(t.dxf.text for t in texts)
            self.assertIn("TOP VIEW", joined)
            self.assertIn("FRONT VIEW", joined)
            self.assertIn("RIGHT VIEW", joined)


if __name__ == "__main__":
    unittest.main()
