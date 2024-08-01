import math
from dataclasses import dataclass
from itertools import combinations_with_replacement as combs_with_rep
from itertools import product
from typing import List, Tuple, Any


@dataclass
class BeaconColorData:
    target_rgb: Tuple[int, int, int] or List[int]
    target_lab: List[float]
    sequence: List[str]
    result_rgb: List[int]
    result_lab: List[float]
    delta_e: float


class BeaconColorCalc:
    colors = {
        0: "white", 1: "lightGray", 2: "gray", 3: "black", 4: "brown", 5: "red", 6: "orange", 7: "yellow", 8: "lime",
        9: "green", 10: "cyan", 11: "lightBlue", 12: "blue", 13: "purple", 14: "magenta", 15: "pink"}
    colors_hex_map = {
        0: 0xf9fffe, 1: 0x9d9d97, 2: 0x474f52, 3: 0x1d1d21, 4: 0x835432, 5: 0xb02e26, 6: 0xf9801d, 7: 0xfed83d,
        8: 0x80c71f, 9: 0x5e7c16, 10: 0x169c9c, 11: 0x3ab3da, 12: 0x3c44aa, 13: 0x8932b8, 14: 0xc74ebd, 15: 0xf38baa}
    _combs = []

    def __init__(self):
        self.color_rgb_map = {k: self._separate_rgb(v) for k, v in self.colors_hex_map.items()}
        self.color_lab_map = {}
        if not self._combs:
            for colors_count in range(1, 7):
                if colors_count > 5:
                    self._combs.extend(combs_with_rep(self.colors.keys(), colors_count))
                else:
                    self._combs.extend(product(self.colors.keys(), repeat=colors_count))

    @staticmethod
    def _string_color_from_id_sequence(seq: list[str]) -> list[str | Any]:
        return [BeaconColorCalc.colors.get(color_id, "Undefined color_id") for color_id in seq]

    @staticmethod
    def _float_rgb_to_integer(rgb: list[float]) -> list[Any]:
        return [math.floor(v * 255) for v in rgb]

    @staticmethod
    def color_string_to_rgb(color: str) -> Tuple[int, int, int]:
        color_int = int(color[1:], 16)
        return (color_int >> 16) & 0xFF, (color_int >> 8) & 0xFF, color_int & 0xFF

    @staticmethod
    def _separate_rgb(rgb: int) -> Tuple[int, int, int]:
        return (rgb & 0xff0000) >> 16, (rgb & 0x00ff00) >> 8, (rgb & 0x0000ff)

    def _rgb2lab(self, rgb: list[int]) -> list[float]:
        if tuple(rgb) in self.color_lab_map:
            return self.color_lab_map[tuple(rgb)]

        r, g, b = [x / 255.0 for x in rgb]

        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

        x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

        x = x ** 0.33333333333 if x > 0.008856 else (7.787 * x) + 0.13793103448
        y = y ** 0.33333333333 if y > 0.008856 else (7.787 * y) + 0.13793103448
        z = z ** 0.33333333333 if z > 0.008856 else (7.787 * z) + 0.13793103448

        lab_values = [116 * y - 16, 500 * (x - y), 200 * (y - z)]

        self.color_lab_map[tuple(rgb)] = lab_values

        return lab_values

    @staticmethod
    def _delta_e1976(lab_a: list[float], lab_b: list[float]) -> float:
        l1, a1, b1 = lab_a
        l2, a2, b2 = lab_b
        return abs(((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2 ** 2)) ** 0.5)

    @staticmethod
    def _delta_e2000(lab_a: list[float], lab_b: list[float]) -> float:
        delta_l = lab_a[0] - lab_b[0]
        delta_a = lab_a[1] - lab_b[1]
        delta_b = lab_a[2] - lab_b[2]
        c1 = math.sqrt(lab_a[1] ** 2 + lab_a[2] ** 2)
        c2 = math.sqrt(lab_b[1] ** 2 + lab_b[2] ** 2)
        delta_c = c1 - c2
        delta_h = delta_a ** 2 + delta_b ** 2 - delta_c ** 2
        delta_h = 0 if delta_h < 0 else delta_h ** 0.5
        sc = 1.0 + 0.045 * c1
        sh = 1.0 + 0.015 * c1
        delta_l_k_l_s_l = delta_l
        delta_c_k_c_s_c = delta_c / sc
        delta_h_k_h_s_h = delta_h / sh
        i = delta_l_k_l_s_l ** 2 + delta_c_k_c_s_c ** 2 + delta_h_k_h_s_h ** 2
        return i ** 0.5

    @staticmethod
    def _create_beacon_color_data(target_rgb, target_lab, best_sequence_str, best_delta_e, result_rgb, result_lab):
        return BeaconColorData(
            target_rgb=target_rgb,
            target_lab=target_lab,
            sequence=best_sequence_str,
            delta_e=best_delta_e,
            result_rgb=result_rgb,
            result_lab=result_lab)

    def _find_best_combination(self, chunk: list[Tuple[str, str, str]], target_lab: list[float], is_accurate: bool):
        min_delta_e = float("inf")
        min_sequence = []
        d_e_func = self._delta_e2000 if is_accurate else self._delta_e1976
        for comb in chunk:
            color = self._sequence_to_color_float_average(comb)
            lab = self._rgb2lab(color)
            delta = d_e_func(lab, target_lab)
            if delta < min_delta_e:
                min_delta_e = delta
                min_sequence = comb
        return min_sequence, min_delta_e

    def color_to_sequence(self, target_rgb, is_accurate: bool = True):
        target_lab = self._rgb2lab(target_rgb)
        best_sequence, best_delta_e = self._find_best_combination(self._combs, target_lab, is_accurate)
        best_sequence_str = self._string_color_from_id_sequence(best_sequence)
        result_rgb = self._sequence_to_color_float_average(best_sequence)
        result_lab = self._rgb2lab(result_rgb)
        data = self._create_beacon_color_data(target_rgb, target_lab, best_sequence_str, best_delta_e, result_rgb,
                                              result_lab)

        return data

    def _sequence_to_color_float_average(self, colors_seq: Tuple[int]) -> List[int]:
        total_r, total_g, total_b = 0.0, 0.0, 0.0

        r, g, b = self.color_rgb_map[colors_seq[0]]
        total_r += r / 255.0
        total_g += g / 255.0
        total_b += b / 255.0

        for i in range(1, len(colors_seq)):
            r, g, b = self.color_rgb_map[colors_seq[i]]
            total_r = (total_r + r / 255.0) / 2.0
            total_g = (total_g + g / 255.0) / 2.0
            total_b = (total_b + b / 255.0) / 2.0
        return self._float_rgb_to_integer([total_r, total_g, total_b])
