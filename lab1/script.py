import argparse
import csv
import math
import os
import sys
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel_1d(size: int, sigma: float) -> list[float]:
    radius = size // 2
    values = [math.exp(-(i * i) / (2 * sigma * sigma)) for i in range(-radius, radius + 1)]
    total = sum(values)
    return [value / total for value in values]


def reflect101(index: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * (length - 1)
    index %= period
    return index if index < length else period - index


def gaussian_blur_native(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    height, width, channels = image.shape
    source = image.tolist()
    weights = gaussian_kernel_1d(size, sigma)
    radius = size // 2

    x_indices = [tuple(reflect101(x + shift, width) for shift in range(-radius, radius + 1))
                 for x in range(width)]
    y_indices = [tuple(reflect101(y + shift, height) for shift in range(-radius, radius + 1))
                 for y in range(height)]

    horizontal = [[[0.0] * channels for _ in range(width)] for _ in range(height)]
    for y in range(height):
        row = source[y]
        for x in range(width):
            sums = horizontal[y][x]
            for weight, xx in zip(weights, x_indices[x]):
                pixel = row[xx]
                for c in range(channels):
                    sums[c] += weight * pixel[c]

    result = [[[0] * channels for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            pixel = result[y][x]
            for weight, yy in zip(weights, y_indices[y]):
                neighbor = horizontal[yy][x]
                for c in range(channels):
                    pixel[c] += weight * neighbor[c]
            for c in range(channels):
                pixel[c] = min(255, max(0, round(pixel[c])))

    return np.asarray(result, dtype=np.uint8)


def gaussian_blur_opencv(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (size, size), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT_101)


def benchmark(function, image: np.ndarray, size: int, sigma: float, repeats: int):
    durations = []
    output = None
    for _ in range(repeats):
        start = perf_counter()
        output = function(image, size, sigma)
        durations.append(perf_counter() - start)
    return output, mean(durations)


def make_comparison_plot(rows: list[dict], destination: Path) -> None:
    sizes = [row["kernel"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, [row["native_ms"] for row in rows], "o-", label="Pure Python")
    ax.plot(sizes, [row["opencv_ms"] for row in rows], "s-", label="OpenCV")
    ax.set_xticks(sizes, [f"{s}×{s}" for s in sizes])
    ax.set_xlabel("Kernel size")
    ax.set_ylabel("Mean execution time (ms)")
    ax.set_title("Gaussian blur: performance by kernel size")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def make_image_grid(image: np.ndarray, results: list[tuple], destination: Path) -> None:
    count = len(results)
    fig, axes = plt.subplots(count + 1, 2, figsize=(10, 4 * (count + 1)), squeeze=False)
    rgb = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(rgb(image))
    axes[0, 0].set_title("Original")
    axes[0, 1].axis("off")
    for row, (size, cv_result, native_result) in enumerate(results, 1):
        for col, (method, processed) in enumerate((("OpenCV", cv_result), ("Pure Python", native_result))):
            axes[row, col].imshow(rgb(processed))
            axes[row, col].set_title(f"{method}, {size}×{size}")
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def make_kernel_plot(kernels: list[int], sigma: float, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in kernels:
        x = range(-(size // 2), size // 2 + 1)
        ax.plot(list(x), gaussian_kernel_1d(size, sigma), marker="o", label=f"{size}×{size}")
    ax.set_xlabel("Distance from center (pixels)")
    ax.set_ylabel("Normalized weight")
    ax.set_title(f"1D Gaussian kernel weights, sigma={sigma}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_difference_map(first: np.ndarray, second: np.ndarray, destination: Path) -> tuple[float, int]:
    delta = cv2.absdiff(first, second)
    gray = np.max(delta, axis=2)
    maximum = int(gray.max())
    mean_difference = float(delta.mean())
    if maximum == 0:
        visual = np.full((max(90, gray.shape[0]), max(350, gray.shape[1]), 3), 35, dtype=np.uint8)
        cv2.putText(visual, "NO DIFFERENCE (0)", (12, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # Автоматическая нормализация: самая большая разница становится яркой.
        contrast = np.rint(gray.astype(np.float32) * (255.0 / maximum)).astype(np.uint8)
        visual = cv2.applyColorMap(contrast, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(destination), visual)
    return mean_difference, maximum


def parse_kernel_sizes(text: str) -> list[int]:
    try:
        sizes = sorted(set(int(part.strip()) for part in text.split(",")))
    except ValueError as error:
        raise ValueError("Размеры ядер должны быть целыми числами через запятую") from error
    if not sizes or any(size < 1 or size % 2 == 0 for size in sizes):
        raise ValueError("Размеры ядер должны быть положительными нечётными числами")
    return sizes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение размеров гауссова ядра: Python и OpenCV")
    parser.add_argument("--input", type=Path, help="Файл изображения; в Google Colab доступна загрузка")
    parser.add_argument("--kernels", default="3,5,7,11,15", help="Нечётные размеры через запятую")
    parser.add_argument("--kernel", type=int, default=None, help="Проверить одно ядро (совместимость со старым запуском)")
    parser.add_argument("--sigma", type=float, default=1.5, help="Стандартное отклонение (> 0), одинаковое для всех ядер")
    parser.add_argument("--repeats", type=int, default=3, help="Число повторений для каждого метода и ядра")
    parser.add_argument("--max-side", type=int, default=256, help="Максимальная сторона изображения; 0 — без уменьшения")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Папка результатов")
    args, _ = parser.parse_known_args(argv)  # игнорируем служебные параметры Jupyter
    try:
        args.kernels = parse_kernel_sizes(str(args.kernel) if args.kernel is not None else args.kernels)
    except ValueError as error:
        parser.error(str(error))
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma должен быть конечным положительным числом")
    if args.repeats < 1 or args.max_side < 0:
        parser.error("--repeats >= 1; --max-side >= 0")
    if args.input is None:
        if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
            from google.colab import files
            print("Загрузите изображение:")
            uploaded = files.upload()
            if not uploaded:
                parser.error("Изображение не загружено")
            args.input = Path(next(iter(uploaded)))
        else:
            parser.error("Укажите --input путь_к_изображению")
    return args


def main(argv=None):
    args = parse_args(argv)
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Не удалось прочитать изображение: {args.input}")
    height, width = image.shape[:2]
    if args.max_side and max(height, width) > args.max_side:
        scale = args.max_side / max(height, width)
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "input.png"), image)
    rows = []
    images = []
    for size in args.kernels:
        cv_result, cv_seconds = benchmark(gaussian_blur_opencv, image, size, args.sigma, args.repeats)
        native_result, native_seconds = benchmark(gaussian_blur_native, image, size, args.sigma, args.repeats)
        difference = cv2.absdiff(cv_result, native_result)
        cv2.imwrite(str(args.output_dir / f"opencv_k{size}.png"), cv_result)
        cv2.imwrite(str(args.output_dir / f"native_k{size}.png"), native_result)
        save_difference_map(cv_result, native_result,
                            args.output_dir / f"methods_difference_k{size}.png")
        row = {"kernel": size, "sigma": args.sigma, "width": image.shape[1],
               "height": image.shape[0], "repeats": args.repeats,
               "opencv_ms": cv_seconds * 1000, "native_ms": native_seconds * 1000,
               "speedup": native_seconds / cv_seconds if cv_seconds > 0 else float("nan"),
               "mae": float(np.mean(difference)), "max_diff": int(np.max(difference))}
        rows.append(row)
        images.append((size, cv_result, native_result))
        print(f"Ядро {size:>2}×{size:<2}: OpenCV {row['opencv_ms']:>9.3f} мс | "
              f"Python {row['native_ms']:>9.3f} мс | x{row['speedup']:>6.1f} | MAE {row['mae']:.4f}")

    # Это отдельное сравнение РАЗНЫХ ЯДЕР: k3 против каждого большего ядра.
    # В отличие от Python vs OpenCV здесь обычно есть реальные изменения изображения.
    base_size, base_cv, _ = images[0]
    for size, cv_result, _ in images[1:]:
        mae, maximum = save_difference_map(
            base_cv, cv_result,
            args.output_dir / f"kernels_difference_k{base_size}_vs_k{size}.png")
        print(f"Ядра {base_size} и {size}: средняя разница {mae:.3f}, максимум {maximum}")

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_comparison_plot(rows, args.output_dir / "timings_by_kernel.png")
    make_image_grid(image, images, args.output_dir / "kernels_comparison.png")
    make_kernel_plot(args.kernels, args.sigma, args.output_dir / "kernel_weights.png")
    print(f"\nРазмер изображения: {image.shape[1]}×{image.shape[0]}; sigma={args.sigma}")
    print(f"Результаты сохранены в: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
