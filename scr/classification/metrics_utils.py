from typing import Dict, Iterable, List


def compute_topk_metrics(predictions: Iterable[dict], ground_truth: Dict[str, int]) -> dict:
    total = 0
    top1_correct = 0
    top5_correct = 0
    missing_ground_truth: List[str] = []

    for item in predictions:
        image = item["image"]
        top5 = item["top5"]

        if image not in ground_truth:
            missing_ground_truth.append(image)
            continue

        label = ground_truth[image]
        total += 1

        if top5 and top5[0] == label:
            top1_correct += 1
        if label in top5:
            top5_correct += 1

    top1 = (100.0 * top1_correct / total) if total else 0.0
    top5 = (100.0 * top5_correct / total) if total else 0.0

    return {
        "evaluated_images": total,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "missing_ground_truth_count": len(missing_ground_truth),
        "missing_ground_truth_preview": missing_ground_truth[:20],
    }
