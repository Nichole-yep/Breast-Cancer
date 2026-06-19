# evaluate/eval_baseline_models.py
# Evaluate baseline segmentation models using the same SegmentationMetrics from evaluate/eval.py.
#
# Put this file into:
# Breast-Cancer/evaluate/eval_baseline_models.py
#
# Example commands:
# python evaluate/eval_baseline_models.py --model unet --weights results/weights/best_unet.pth --csv_file preprocess/test.csv --device cpu
# python evaluate/eval_baseline_models.py --model attention_unet --weights results/weights/best_attention_unet.pth --csv_file preprocess/test.csv --device cpu
# python evaluate/eval_baseline_models.py --model deeplabv3plus --weights results/weights/best_deeplabv3plus.pth --csv_file preprocess/test.csv --device cpu

import os
import argparse
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocess.dataset import BUSIDataset
from evaluate.eval import SegmentationMetrics
from models.baseline_models import get_baseline_model


def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        return model_output[-1]
    if isinstance(model_output, dict):
        if "out" in model_output:
            return model_output["out"]
        return list(model_output.values())[-1]
    return model_output


def evaluate_model(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Same preprocess module as training. augment=False for fair evaluation.
    dataset = BUSIDataset(
        csv_file=args.csv_file,
        ues_lee=(not args.no_lee),
        ues_clahe=(not args.no_clahe),
        augment=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Test samples: {len(dataset)}")

    model = get_baseline_model(
        model_name=args.model,
        in_channels=3,
        num_classes=1,
        base_channels=args.base_channels
    ).to(device)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weight file not found: {args.weights}")

    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    metrics = SegmentationMetrics(num_classes=2)

    with torch.no_grad():
        for images, masks, _edges in tqdm(dataloader, desc=f"Evaluating {args.model}"):
            images = images.to(device)
            masks = masks.to(device).float()

            outputs = model(images)
            logits = extract_logits(outputs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = (torch.sigmoid(logits) > args.threshold).long()

            preds_np = preds.squeeze(1).cpu().numpy().astype("uint8")
            masks_np = (masks.squeeze(1).cpu().numpy() > 0).astype("uint8")

            for i in range(preds_np.shape[0]):
                metrics.update_with_boundary(preds_np[i], masks_np[i])

    scores = metrics.get_scores()

    print("\n" + "=" * 70)
    print(f"Baseline Evaluation Result: {args.model}")
    print("=" * 70)
    print(f"Dice:                  {scores['dice']:.4f}")
    print(f"IoU / Jaccard:          {scores['iou']:.4f}")
    print(f"mIoU:                  {scores['miou']:.4f}")
    print(f"Accuracy:              {scores['accuracy']:.4f}")
    print(f"Precision:             {scores['precision']:.4f}")
    print(f"Sensitivity / Recall:  {scores['sensitivity']:.4f}")
    print(f"Specificity:           {scores['specificity']:.4f}")
    print(f"HD95 mean ± std:       {scores['hd95_mean']:.2f} ± {scores['hd95_std']:.2f}")
    print(f"Valid HD95 samples:    {scores['valid_hd95_count']}")
    print(f"Confusion matrix foreground: TP={scores['TP']}, FP={scores['FP']}, FN={scores['FN']}, TN={scores['TN']}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    result_csv = os.path.join(args.output_dir, f"{args.model}_test_metrics.csv")
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dice", "iou", "miou", "accuracy", "precision", "sensitivity", "specificity", "hd95_mean", "hd95_std", "valid_hd95_count", "TP", "FP", "FN", "TN"])
        writer.writerow([
            args.model,
            f"{scores['dice']:.6f}",
            f"{scores['iou']:.6f}",
            f"{scores['miou']:.6f}",
            f"{scores['accuracy']:.6f}",
            f"{scores['precision']:.6f}",
            f"{scores['sensitivity']:.6f}",
            f"{scores['specificity']:.6f}",
            f"{scores['hd95_mean']:.6f}",
            f"{scores['hd95_std']:.6f}",
            scores['valid_hd95_count'],
            scores['TP'],
            scores['FP'],
            scores['FN'],
            scores['TN']
        ])
    print(f"Saved metrics CSV to: {result_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline model with the group's evaluate/eval.py metrics.")

    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "attention_unet", "deeplabv3plus"])
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained baseline weights, e.g. results/weights/best_unet.pth")
    parser.add_argument("--csv_file", type=str, default="preprocess/test.csv")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results/baseline_metrics")

    parser.add_argument("--no_lee", action="store_true", help="Disable Lee filtering in BUSIDataset if supported.")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE in BUSIDataset if supported.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)
