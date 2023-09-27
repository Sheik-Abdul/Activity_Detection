# Activity_Detection
An android app that collects data from a wearable device and predicts the activity done by the user locally. The trained model is uploaded to a federated learning server to aggregate the models from multiple users to improve model efficiency and personal data protection.

# Project tree

 * [MachineLearning](./MachineLearning)
   * [optuna.ipynb](./MachineLearning/optuna.ipynb)
   * [random_search.ipynb](./MachineLearning/random_search.ipynb)
 * [FederatedLearning](./FederatedLearning)
   * [FL.py](./FederatedLearning/FL.py)
   * [client.py](./FederatedLearning/client.py)
   * [client2.py](./FederatedLearning/client2.py)
 * [README.md](./README.md)
