'''
This script is used to run the evaluation from the command line.
It takes the true labels and predicted labels as input,
and outputs the evaluation metrics in a JSON file.

Run it using for example:
ml-eval --y_true tests/y_true.csv --y_pred tests/y_pred.csv --output results/
'''

import argparse
from ml_eval.metrics import calculate_metrics
import json
import os

def main():
    print("\nStarted ml-eval-kit evalutation.\n")

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_true", type=str, required=True, help="Path to a .csv file with the true labels")
    parser.add_argument("--y_pred", type=str, required=True, help="Path to a .csv file with the predicted labels")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save the evaluation metrics")
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
    

    print("Evaluation completed.\n\nResults saved to:", f"{os.path.abspath(args.output)}\metrics.json\n")

if __name__ == "__main__":
    main()
