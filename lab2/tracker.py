import argparse
import csv
import sys
import time

import cv2
import numpy as np


def parse_roi(value: str):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI задаётся как x,y,w,h")
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("w и h должны быть > 0")
    return x, y, w, h


def is_plausible(quad: np.ndarray, frame_area: float) -> bool:
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return False
    area = cv2.contourArea(quad.astype(np.float32))
    return 0.001 * frame_area < area < 4.0 * frame_area


def locate(gray, sift, matcher, kp_t, des_t, corners, args, frame_area):

    kp_f, des_f = sift.detectAndCompute(gray, None)
    if des_f is None or len(kp_f) < 2:
        return None, 0, 0

    pairs = matcher.knnMatch(des_t, des_f, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < args.ratio * n.distance]
    if len(good) < args.min_matches:
        return None, len(good), 0

    src = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, args.ransac_thresh)
    if H is None:
        return None, len(good), 0

    inliers = int(mask.sum())
    if inliers < args.min_inliers:
        return None, len(good), inliers

    quad = cv2.perspectiveTransform(corners, H)
    if not is_plausible(quad, frame_area):
        return None, len(good), inliers
    return quad, len(good), inliers


def draw(frame, quad, label, inliers):
    pts = np.int32(quad).reshape(-1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    x, y = pts[np.argmin(pts[:, 1])]
    text = f"{label} ({inliers} inl)"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    y = max(y, th + 10)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (0, 255, 0), cv2.FILLED)
    cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Трекинг плоского объекта по ключевым точкам SIFT")
    ap.add_argument("input", help="входное видео")
    ap.add_argument("-o", "--output", default="out.mp4", help="выходное видео")
    ap.add_argument("--nfeatures", type=int, default=3000, help="максимум точек SIFT на изображении")
    ap.add_argument("--ratio", type=float, default=0.75, help="порог ratio-теста Лоу")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--ransac-thresh", type=float, default=5.0, help="порог репроекции RANSAC, px")
    ap.add_argument("--roi", type=parse_roi, help="область объекта на 1-м кадре: x,y,w,h")
    ap.add_argument("--select-roi", action="store_true", help="выделить объект мышью на 1-м кадре")
    ap.add_argument("--label", default="Object", help="подпись рамки (только латиница — ограничение putText)")
    ap.add_argument("--show", action="store_true", help="показывать окно во время обработки (q — выход)")
    ap.add_argument("--csv", help="покадровая статистика в CSV")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.input}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр")

    fh, fw = first.shape[:2]
    if args.select_roi:
        x, y, w, h = cv2.selectROI("Select object", first, showCrosshair=True)
        cv2.destroyWindow("Select object")
        if w == 0 or h == 0:
            raise RuntimeError("Область объекта не выбрана")
    elif args.roi:
        x, y, w, h = args.roi
    else:
        x, y, w, h = 0, 0, fw, fh
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        raise ValueError(f"ROI {x},{y},{w},{h} выходит за кадр {fw}x{fh}")

    sift = cv2.SIFT_create(nfeatures=args.nfeatures)
    # FLANN с KD-деревьями: приближённый быстрый поиск ближайших 128-мерных векторов
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    template = cv2.cvtColor(first[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    kp_t, des_t = sift.detectAndCompute(template, None)
    if des_t is None or len(kp_t) < args.min_matches:
        raise RuntimeError(f"На эталоне слишком мало ключевых точек ({0 if des_t is None else len(kp_t)}). "
                           "Нужен более текстурный объект.")
    print(f"Эталон {w}x{h}, ключевых точек SIFT: {len(kp_t)}")
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("ВНИМАНИЕ: FPS не указан в метаданных видео, пишу выход с 30 FPS", file=sys.stderr)
        fps = 30.0
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть на запись: {args.output}")

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["frame", "found", "good_matches", "inliers", "ms"])

    frame_area = float(fw * fh)
    total = found = 0
    inliers_sum = 0
    time_sum = 0.0
    frame = first

    while frame is not None:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        quad, n_good, n_inl = locate(gray, sift, matcher, kp_t, des_t, corners, args, frame_area)
        ms = (time.perf_counter() - t0) * 1000.0

        if quad is not None:
            draw(frame, quad, args.label, n_inl)
            found += 1
            inliers_sum += n_inl
        else:
            cv2.putText(frame, "NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        if csv_writer:
            csv_writer.writerow([total, int(quad is not None), n_good, n_inl, f"{ms:.1f}"])
        total += 1
        time_sum += ms

        if args.show:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Прервано пользователем")
                break

        ok, frame = cap.read()
        if not ok:
            frame = None

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Кадров: {total}, объект найден: {found} ({100.0 * found / total:.1f}%)")
    if found:
        print(f"Среднее число инлаеров: {inliers_sum / found:.1f}")
    print(f"Среднее время на кадр: {time_sum / total:.1f} мс ({1000.0 * total / time_sum:.1f} FPS)")
    print(f"Результат: {args.output}")


if __name__ == "__main__":
    main()
