import argparse
from ml_eval.metrics import calculate_metrics
import json
import os

def main():
    print("Started ml-eval-kit evalutation")

    parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline for BRATS Challenge")
    parser.add_argument("--y_true", type=str, required=True, help="Path to a .csv file with the true labels")
    parser.add_argument("--y_pred", type=str, required=True, help="Path to a .csv file with the predicted labels")
    parser.add_argument("--output", type=str, required=True, help="Path to save the evaluation results")

    args = parser.parse_args()

    # Load the true labels and predictions from the provided paths
    with open(args.y_true, 'r') as f:
        labels = []
        for line in f.readlines():
            labels.append(int(line.strip().rstrip(",")))
    
    with open(args.y_pred, 'r') as f:
        predic = []
        for line in f.readlines():
            predic.append(int(line.strip().rstrip(",")))
    

    metrics = calculate_metrics(labels, predic)

    # If the output directory does not exist, create it
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Save the metrics to a JSON file
    with open(args.output+"/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    

    print("Evaluation completed. Results saved to:", args.output + "metrics.json")


if __name__ == "__main__":
    main()
