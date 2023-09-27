import flwr as fl
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Client timeout duration in seconds
client_timeout = 10

# List to store aggregated accuracy values
aggregated_f1_score = []


def evaluate(client):
    # Set a timeout for participant responses
    with fl.time_limit(client_timeout):
        # Evaluate the client and handle any failures
        try:
            return client.evaluate()
        except fl.common.FLTimeoutError:
            # Handle participant failure
            print("Client failed to respond within the timeout threshold.")


# Define the aggregation function
def aggregate_metrics(metrics_list):
    global aggregated_f1_score   # Reference the global variable

    metrics_list = list(metrics_list)
    print("Metrics List:")
    print(metrics_list)
    # Get F1 score from clients
    f1_scores = []
    for _, metrics in metrics_list:
        if "F1 Score" in metrics:
            f1_scores.append(metrics["F1 Score"])
        else:
            print("Metrics does not have F1 score key:", metrics)
    # Aggregate F1 score from all clients
    agg_f1_score = sum(f1_scores) / len(f1_scores)

    print("Aggregated F1 Score: ", agg_f1_score)   # Print aggregated F1 score

    # Append the aggregated F1 score to the global list
    aggregated_f1_score.append(agg_f1_score)

    # Store aggregated metrics in dictionary
    aggregated_metrics = {"F1 Score": agg_f1_score}  # Store metrics for future references

    # Save aggregated F1 score to a file
    with open('aggregated_f1_score.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([agg_f1_score])

    return aggregated_metrics


# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    min_fit_clients=1,  # Never sample less than 1 clients for training
    min_evaluate_clients=1,  # Never sample less than 1 clients for evaluation
    min_available_clients=2,  # Wait until all 2 clients are available
    evaluate_metrics_aggregation_fn=aggregate_metrics  # custom aggregation function
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024 * 1024 * 1024,
    strategy=strategy
)

# Perform clustering and plotting after the server finishes
# Wait for the server to finish
time.sleep(300)

# Perform clustering and plotting after the server finishes
# Perform clustering of aggregated f1 score
num_clusters = 2
if len(aggregated_f1_score) > 0:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(aggregated_f1_score).reshape(-1, 1))
    cluster_labels = kmeans.labels_

    # Create a scatter plot of the clustered federated learning results
    plt.scatter(range(1, len(aggregated_f1_score) + 1), aggregated_f1_score, c=cluster_labels)
    plt.xlabel("Round")
    plt.ylabel("Aggregated F1 Score")
    plt.title("Clustered Federated Learning Results")

    # Save the plot to a file
    plt.savefig("clustered_federated_learning.png")

    # Show the plot
    plt.show()
else:
    print("No aggregated f1 score available for clustering.")
