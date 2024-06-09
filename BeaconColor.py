import math
import multiprocessing as mp
from dataclasses import dataclass
from itertools import combinations_with_replacement as combs_with_rep
from itertools import product
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BeaconColorData:
    target_rgb: Tuple[int, int, int] or list[int, int, int]
    target_lab: Tuple[int, int, int] or list[int, int, int]
    sequence: list[str, ...] or Tuple[str, ...]
    result_rgb: Tuple[int, int, int] or list[int, int, int]
    result_lab: Tuple[int, int, int] or list[int, int, int]
    delta_e: int


class BeaconColorCalc:
    colors = {
        0: "white", 1: "lightGray", 2: "gray", 3: "black", 4: "brown", 5: "red", 6: "orange", 7: "yellow", 8: "lime",
        9: "green", 10: "cyan", 11: "lightBlue", 12: "blue", 13: "purple", 14: "magenta", 15: "pink"}
    colors_hex_map = {
        0: 0xf9fffe, 1: 0x9d9d97, 2: 0x474f52, 3: 0x1d1d21, 4: 0x835432, 5: 0xb02e26, 6: 0xf9801d, 7: 0xfed83d,
        8: 0x80c71f, 9: 0x5e7c16, 10: 0x169c9c, 11: 0x3ab3da, 12: 0x3c44aa, 13: 0x8932b8, 14: 0xc74ebd, 15: 0xf38baa}

    def __init__(self):
        self.color_rgb_map = {k: self._separate_rgb(v) for k, v in self.colors_hex_map.items()}
        self.color_lab_map = {}
        self.combs = []
        for colors_count in range(1, 7):
            if colors_count > 4:
                self.combs.extend(combs_with_rep(self.colors.keys(), colors_count))
            else:
                self.combs.extend(product(self.colors.keys(), repeat=colors_count))

    @staticmethod
    def _string_color_from_id_sequence(seq: list[int, ...]) -> list[str | Any]:
        return [BeaconColorCalc.colors.get(color_id, "Undefined color_id") for color_id in seq]

    @staticmethod
    def _chunkify(lst: List[Any], n: int) -> List[List[Tuple[str]]]:
        return [lst[i::n] for i in range(n)]

    @staticmethod
    def color_string_to_rgb(color: str) -> Tuple[int, int, int]:
        return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    @staticmethod
    def _float_rgb_to_integer(rgb: list[float, float, float]) -> list[Any]:
        return [math.floor(v * 255) for v in rgb]

    @staticmethod
    def _separate_rgb(rgb: int) -> Tuple[int, int, int]:
        return (rgb & 0xff0000) >> 16, (rgb & 0x00ff00) >> 8, (rgb & 0x0000ff)

    def _rgb2lab(self, rgb: list[int, int, int]) -> List[float]:
        # Проверяем, были ли уже вычислены значения Lab для данного RGB цвета
        if tuple(rgb) in self.color_lab_map:
            return self.color_lab_map[tuple(rgb)]

        r, g, b = [x / 255.0 for x in rgb]

        r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
        g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
        b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

        x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
        y = r * 0.2126 + g * 0.7152 + b * 0.0722
        z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

        x = x ** (1 / 3) if x > 0.008856 else (7.787 * x) + (16 / 116)
        y = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)
        z = z ** (1 / 3) if z > 0.008856 else (7.787 * z) + (16 / 116)

        lab_values = [116 * y - 16, 500 * (x - y), 200 * (y - z)]

        # Кешируем вычисленные значения Lab для данного RGB цвета
        self.color_lab_map[tuple(rgb)] = lab_values

        return lab_values

    @staticmethod
    def _delta_e(lab_a: list[float, float, float], lab_b: list[float, float, float]) -> float:
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
        delta_l_k_l_s_l = delta_l / 1.0
        delta_c_k_c_s_c = delta_c / sc
        delta_h_k_h_s_h = delta_h / sh
        i = delta_l_k_l_s_l ** 2 + delta_c_k_c_s_c ** 2 + delta_h_k_h_s_h ** 2
        return 0 if i < 0 else math.sqrt(i)

    @staticmethod
    def create_beacon_color_data(target_rgb, target_lab, best_sequence_str, best_delta_e, result_rgb, result_lab):
        return BeaconColorData(
            target_rgb=target_rgb,
            target_lab=target_lab,
            sequence=best_sequence_str,
            delta_e=best_delta_e,
            result_rgb=result_rgb,
            result_lab=result_lab)

    def _find_best_combination(self, chunk: list[Tuple[str, ...]], target_lab: list[float, float, float]):
        min_delta_e = float("inf")
        min_sequence = []
        for comb in chunk:
            color = self._sequence_to_color_float_average(comb)
            lab = self._rgb2lab(color)
            delta = self._delta_e(lab, target_lab)
            if delta < min_delta_e:
                min_delta_e = delta
                min_sequence = comb
        return min_sequence, min_delta_e

    def color_to_sequence(self, target_rgb):
        target_lab = self._rgb2lab(target_rgb)
        best_sequence, best_delta_e = self._find_best_combination(self.combs, target_lab)
        best_sequence_str = self._string_color_from_id_sequence(best_sequence)
        result_rgb = self._sequence_to_color_float_average(best_sequence)
        result_lab = self._rgb2lab(result_rgb)
        data = self.create_beacon_color_data(target_rgb, target_lab, best_sequence_str, best_delta_e, result_rgb,
                                             result_lab)

        return data

    def color_to_sequence_parallel(self, target_rgb, num_processes=mp.cpu_count()):
        target_lab = self._rgb2lab(target_rgb)

        chunks = self._chunkify(self.combs, num_processes)
        pool = mp.Pool(processes=num_processes)

        results = [pool.apply_async(self._find_best_combination, args=(chunk, target_lab)) for chunk in chunks]
        pool.close()
        pool.join()

        best_delta_e = float("inf")
        best_sequence = []
        for result in results:
            sequence, delta_e_value = result.get()
            if delta_e_value < best_delta_e:
                best_delta_e = delta_e_value
                best_sequence = sequence

        best_sequence_str = self._string_color_from_id_sequence(best_sequence)
        result_rgb = self._sequence_to_color_float_average(best_sequence)
        result_lab = self._rgb2lab(result_rgb)
        data = self.create_beacon_color_data(target_rgb, target_lab, best_sequence_str, best_delta_e, result_rgb,
                                             result_lab)

        return data

    def _sequence_to_color_float_average(self, color_str: Tuple[str]) -> List[int]:
        total_r, total_g, total_b = 0, 0, 0
        count = len(color_str)

        r, g, b = self.color_rgb_map[color_str[0]]
        total_r += r
        total_g += g
        total_b += b

        for i in range(1, count):
            r, g, b = self.color_rgb_map[color_str[i]]
            total_r += r
            total_g += g
            total_b += b

        avg_r, avg_g, avg_b = total_r / (count * 255), total_g / (count * 255), total_b / (count * 255)
        return self._float_rgb_to_integer([avg_r, avg_g, avg_b])

    @staticmethod
    def create_color_image(color_left: Tuple[int, int, int] or List[int, int, int],
                           color_right: Tuple[int, int, int] or List[int, int, int], width=200, height=100):
        color_left = tuple(c / 255 for c in color_left)
        color_right = tuple(c / 255 for c in color_right)
        left_half = np.full((height, width // 2, 3), color_left)
        right_half = np.full((height, width // 2, 3), color_right)
        image = np.concatenate((left_half, right_half), axis=1)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
