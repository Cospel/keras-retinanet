import warnings
import csv
import argparse
import sys
import os

import numpy as np
import scipy.optimize

from PIL import Image

from keras_retinanet.preprocessing.csv_generator import _open_for_csv
from keras_retinanet.utils.anchors import generate_anchors, AnchorParameters, anchors_for_shape, compute_overlap
from keras_retinanet.utils.image import compute_resize_scale, compute_resize_fixed

warnings.simplefilter("ignore")

SIZES = [32, 64, 128, 256, 512]
STRIDES = [8, 16, 32, 64, 128]
state = {"best_result": sys.maxsize}


def calculate_config(values, ratio_count):
    split_point = int((ratio_count - 1) / 2)

    ratios = [1]
    for i in range(split_point):
        ratios.append(values[i])
        ratios.append(1 / values[i])

    scales = values[split_point:]

    return AnchorParameters(SIZES, STRIDES, ratios, scales)


def base_anchors_for_shape(pyramid_levels=None, anchor_params=None):
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx], ratios=anchor_params.ratios, scales=anchor_params.scales
        )
        all_anchors = np.append(all_anchors, anchors, axis=0)

    return all_anchors


def average_overlap(values, entries, state, image_shape, mode="focal", ratio_count=3, include_stride=False):
    anchor_params = calculate_config(values, ratio_count)

    if include_stride:
        anchors = anchors_for_shape(image_shape, anchor_params=anchor_params)
    else:
        anchors = base_anchors_for_shape(anchor_params=anchor_params)

    overlap = compute_overlap(entries, anchors)
    max_overlap = np.amax(overlap, axis=1)
    not_matched = len(np.where(max_overlap < 0.5)[0])

    if mode == "avg":
        result = 1 - np.average(max_overlap)
    elif mode == "ce":
        result = np.average(-np.log(max_overlap))
    elif mode == "focal":
        result = np.average(-(1 - max_overlap) ** 2 * np.log(max_overlap))
    else:
        raise Exception("Invalid mode.")

    if "iteration" not in state:
        state["iteration"] = 0
    else:
        state["iteration"] += 1

    if result < state["best_result"]:
        state["best_result"] = result

        print(f"Current best anchor configuration {result}", state["iteration"])
        print(f"Ratios: {sorted(np.round(anchor_params.ratios, 3))}")
        print(f"Scales: {sorted(np.round(anchor_params.scales, 3))}")

        if include_stride:
            print(f"Average overlap: {np.round(np.average(max_overlap), 3)}")

        print(f"Number of labels that don't have any matching anchor: {not_matched}")
        print()

    return result, not_matched


class AnchorOptimizer:
    def __init__(
        self,
        images,
        scales: int = 3,
        ratios: int = 3,
        include_stride: bool = False,
        objective: str = "focal",
        popsize: int = 100,
        resize: bool = False,
        image_min_side: int = 800,
        image_max_side: int = 800,
        seed: int = 2,
    ):
        if ratios % 2 != 1:
            raise Exception("The number of ratios has to be odd.")

        entries = np.zeros((0, 4))
        max_x = 0
        max_y = 0

        if seed:
            seed = np.random.RandomState(seed)
        else:
            seed = np.random.RandomState()

        print("Loading object dimensions.")

        for image in images:
            for o in range(len(image["_objects"])):
                x1, y1, x2, y2 = image["_objects"][o]["bound_box"]
                
                if resize:
                    print(image)
                    image_shape = (image["height"], image["width"], 3) # todo check
                    scale_h, scale_w = compute_resize_fixed(image_shape, image_min_side, image_max_side)
                    x1 *= scale_w
                    y1 *= scale_h
                    x2 *= scale_w
                    y2 *= scale_h

                max_x = max(x2, max_x)
                max_y = max(y2, max_y)

            if include_stride:
                entry = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
                entries = np.append(entries, entry, axis=0)
            else:
                width = x2 - x1
                height = y2 - y1
                entry = np.expand_dims(np.array([-width / 2, -height / 2, width / 2, height / 2]), axis=0)
                entries = np.append(entries, entry, axis=0)

            print("image", image)

        image_shape = [max_y, max_x]

        print("Optimising anchors.")

        bounds = []
        best_result = sys.maxsize

        for i in range(int((ratios - 1) / 2)):
            bounds.append((1, 4))

        for i in range(scales):
            bounds.append((0.4, 2))

        print(f"Starting to optimize popsize {popsize}, heighXwidth ({image_min_side}x{image_max_side})")
        result = scipy.optimize.differential_evolution(
            lambda x: average_overlap(x, entries, state, image_shape, objective, ratios, include_stride)[0],
            bounds=bounds,
            popsize=popsize,
            seed=seed,
        )

        if hasattr(result, "success") and result.success:
            print("Optimization ended successfully!")
        elif not hasattr(result, "success"):
            print("Optimization ended!")
        else:
            print("Optimization ended unsuccessfully!")
            print(f"Reason: {result.message}")

        values = result.x
        anchor_params = calculate_config(values, ratios)
        (avg, not_matched) = average_overlap(
            values, entries, {"best_result": 0}, image_shape, "avg", ratios, include_stride
        )

        print()
        print("Final best anchor configuration")
        print(f"Ratios: {sorted(np.round(anchor_params.ratios, 3))}")
        print(f"Scales: {sorted(np.round(anchor_params.scales, 3))}")

        if include_stride:
            print(f"Average overlap: {np.round(1 - avg, 3)}")

        self.ratios = sorted(np.round(anchor_params.ratios, 3))
        self.scales = sorted(np.round(anchor_params.scales, 3))

        print(f"Number of labels that don't have any matching anchor: {not_matched}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize RetinaNet anchor configuration")
    parser.add_argument("annotations", help="Path to CSV file containing annotations for anchor optimization.")
    parser.add_argument("--scales", type=int, help="Number of scales.", default=3)
    parser.add_argument("--ratios", type=int, help="Number of ratios, has to be an odd number.", default=3)
    parser.add_argument(
        "--include-stride",
        action="store_true",
        help="Should stride of the anchors be taken into account. Setting this to false will give "
        "more accurate results however it is much slower.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="focal",
        help="Function used to weight the difference between the target and proposed anchors. "
        "Options: focal, avg, ce.",
    )
    parser.add_argument(
        "--popsize", type=int, default=15, help="The total population size multiplier used by differential evolution."
    )
    parser.add_argument("--no-resize", help="Disable image resizing.", dest="resize", action="store_false")
    parser.add_argument(
        "--image-min-side", help="Rescale the image so the smallest side is min_side.", type=int, default=800
    )
    parser.add_argument(
        "--image-max-side",
        help="Rescale the image if the largest side is larger than max_side.",
        type=int,
        default=1333,
    )
    parser.add_argument("--seed", type=int, help="Seed value to use for differential evolution.")
    args = parser.parse_args()

    if args.ratios % 2 != 1:
        raise Exception("The number of ratios has to be odd.")

    entries = np.zeros((0, 4))
    max_x = 0
    max_y = 0

    if args.seed:
        seed = np.random.RandomState(args.seed)
    else:
        seed = np.random.RandomState()

    print("Loading object dimensions.")

    with _open_for_csv(args.annotations) as file:
        for line, row in enumerate(csv.reader(file, delimiter=",")):
            x1, y1, x2, y2 = list(map(lambda x: int(x), row[1:5]))

            if not x1 or not y1 or not x2 or not y2:
                continue

            if args.resize:
                # Concat base path from annotations file follow retinanet
                base_dir = os.path.split(args.annotations)[0]
                relative_path = row[0]
                image_path = os.path.join(base_dir, relative_path)
                img = Image.open(image_path)

                if hasattr(img, "shape"):
                    image_shape = img.shape
                else:
                    image_shape = (img.size[0], img.size[1], 3)

                scale = compute_resize_scale(image_shape, min_side=args.image_min_side, max_side=args.image_max_side)
                x1, y1, x2, y2 = list(map(lambda x: int(x) * scale, row[1:5]))

            max_x = max(x2, max_x)
            max_y = max(y2, max_y)

            if args.include_stride:
                entry = np.expand_dims(np.array([x1, y1, x2, y2]), axis=0)
                entries = np.append(entries, entry, axis=0)
            else:
                width = x2 - x1
                height = y2 - y1
                entry = np.expand_dims(np.array([-width / 2, -height / 2, width / 2, height / 2]), axis=0)
                entries = np.append(entries, entry, axis=0)

    image_shape = [max_y, max_x]

    print("Optimising anchors.")

    bounds = []
    best_result = sys.maxsize

    for i in range(int((args.ratios - 1) / 2)):
        bounds.append((1, 4))

    for i in range(args.scales):
        bounds.append((0.4, 2))

    result = scipy.optimize.differential_evolution(
        lambda x: average_overlap(x, entries, state, image_shape, args.objective, args.ratios, args.include_stride)[0],
        bounds=bounds,
        popsize=args.popsize,
        seed=seed,
        maxiter=100
    )

    if hasattr(result, "success") and result.success:
        print("Optimization ended successfully!")
    elif not hasattr(result, "success"):
        print("Optimization ended!")
    else:
        print("Optimization ended unsuccessfully!")
        print(f"Reason: {result.message}")

    values = result.x
    anchor_params = calculate_config(values, args.ratios)
    (avg, not_matched) = average_overlap(
        values, entries, {"best_result": 0}, image_shape, "avg", args.ratios, args.include_stride
    )

    print()
    print("Final best anchor configuration")
    print(f"Ratios: {sorted(np.round(anchor_params.ratios, 3))}")
    print(f"Scales: {sorted(np.round(anchor_params.scales, 3))}")

    if args.include_stride:
        print(f"Average overlap: {np.round(1 - avg, 3)}")

    print(f"Number of labels that don't have any matching anchor: {not_matched}")
