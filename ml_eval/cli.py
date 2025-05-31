'''
This script is used to run the evaluation from the command line.
It takes the true labels and predicted labels as input,
and outputs the evaluation metrics in a JSON file.

Run it using for example:
ml-eval --y_true tests/y_true.csv --y_pred tests/y_pred.csv --output results/
'''

# Calculating metrics
import argparse
from ml_eval.metrics import calculate_metrics
import json

# Other imports
import os

# Creating confusion matrix
from ml_eval.plots import get_conf_matrix, create_conf_matrix_fig

def main():
    print("\nStarted ml-eval-kit evalutation...", end="\r")

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--y_true", type=str, required=True, help="Path to a .csv file with the true labels")
    parser.add_argument("--y_pred", type=str, required=True, help="Path to a .csv file with the predicted labels")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save the evaluation metrics")
    parser.add_argument("--conf_matrix_title", type=str, help="Title for the confusion matrix figure", default="Confusion Matrix")
    parser.add_argument("--conf_matrix_subtitle", type=str, help="Subtitle for the confusion matrix figure", default=None)
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
    
    # Create confusion matrix
    matrix = get_conf_matrix(predic, labels)
    create_conf_matrix_fig(matrix, save_fig_as=args.output+"/confusion_matrix.png", title=args.conf_matrix_title, subtitle=args.conf_matrix_subtitle)

    print("Started ml-eval-kit evalutation.       \nEvaluation completed.")
    print("\nResults saved to:", f"{os.path.abspath(args.output)}\\metrics.json\n")

if __name__ == "__main__":
    main()
