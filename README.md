
# Universal Anomaly Detection and Predictive Modeling Framework

This project provides a versatile framework for anomaly detection and predictive modeling across diverse datasets and domains. It includes implementations of various machine learning and statistical algorithms, enabling efficient data analysis, anomaly detection, and forecasting tasks.

## Supported Algorithms for Anomaly Detection

This table lists the available anomaly detection algorithms, categorized by type.
```
| **Category**              | **Algorithm Name**          | **Description**                                                                 |
|---------------------------|-----------------------------|---------------------------------------------------------------------------------|
| **Statistical**           | ARIMA                      | A statistical model used for time series forecasting and anomaly detection.     |
|                           | Mahalanobis Distance       | Detects anomalies based on the Mahalanobis distance metric.                     |
|                           | SH-ESD                     | Seasonal Hybrid Extreme Studentized Deviate method for anomaly detection.       |
| **Machine Learning**      | Isolation Forest           | Anomaly detection using the Isolation Forest algorithm.                         |
|                           | K-Nearest Neighbors (KNN)  | Detects anomalies based on distance from nearest neighbors.                     |
|                           | One-Class SVM              | A variant of SVM used for anomaly detection in single-class datasets.           |
|                           | Random Cut Forest          | A tree-based model for anomaly detection in time series data.                   |
|                           | XGBoost                    | A gradient boosting algorithm used for detecting anomalies in structured data.   |
|                           | AdaBoost                   | A boosting algorithm used for detecting anomalies in data.                      |
| **Clustering**            | K-Means                   | A clustering algorithm to identify outliers based on cluster distance.          |
|                           | DBSCAN                     | Density-based spatial clustering algorithm for anomaly detection.               |
|                           | Local Outlier Factor (LOF) | Identifies outliers by measuring the local density deviation.                   |
| **Deep Learning**         | AutoEncoder               | A neural network used to detect anomalies by learning a compact representation. |
|                           | LSTM Autoencoder           | Combines LSTM and Autoencoder for time series anomaly detection.                |
|                           | VAE                        | Variational Autoencoder used for unsupervised anomaly detection.                |
|                           | DeepAD                     | A deep learning-based anomaly detection method.                                 |
|                           | TimeGAN                    | A generative adversarial network designed for time series anomaly detection.     |
|                           | Deep Autoencoder           | A deeper variant of Autoencoder for complex anomaly detection.                  |
| **Probabilistic Models**  | Gaussian Process           | Uses probabilistic modeling to detect anomalies in time series.                 |
|                           | Bayesian Network           | A probabilistic graphical model for anomaly detection.                          |
| **Time Series**           | Prophet                    | Forecasting algorithm used for time series anomaly detection.                   |
|                           | Matrix Profile             | Uses matrix profiles for time series anomaly detection.                         |
|                           | DTW                        | Dynamic Time Warping used for anomaly detection in time series.                 |
| **Dimensionality Reduction** | Robust PCA              | A robust principal component analysis method for anomaly detection.             |
```

## Project Structure

metro_air_leak_anomaly_detection/
```
├── raw_data/                   # Raw data files (e.g., original CSV files)
├── src/                        # Source code for the project
│   ├── init.py                 # Init file for package recognition
│   ├── data                    # Data loading and preprocessing functions
│   │   ├── init.py
│   │   ├── loader.py           # load data from postgres or somewhere else
│   │   ├── preprocessing.py    # preprocessing steps before training a model
│   ├── evaluation              # benchmarking the models performances
│   │   ├── init.py
│   │   ├── metrices.py         # calculate the performance metrices
│   ├── models/                 # Folder for model classes
│   │   ├── init.py
│   │   ├── base_model.py       # base_model implementation which will be inherited by all models 
│   │   ├── {model_name}.py     # specific model implementation
│   ├── results                 # trained models will be saved here as well as performance report
│   │   ├── {batch_number}_datetime       # ex: 001_2024-11-19_17-03 : formated way to store the models and reports
│   │   │   ├── models          # store all trained models as '.pkl' fromat
│   │   │   │   ├── {model_name}_model.pkl          # trained model as '.pkl' fromat
│   │   │   ├── reports.csv     # performance metrices of all trained models as csv file
│   ├── utils.py                # helper functions
│   │   └── {file_name}.py      # helper functions to reuse and keep the main functions clean
│   ├── visualization           # data visualization
│   │   ├── init.py
│   │   └── plotter.py          # class to generate graph/plot
│   ├── generate_plots.py       # Main script to execute end-to-end model data plot/graph
│   └── train_models.py         # Main script to execute end-to-end model training
├── tests/                      # Unit tests and integration tests
├── scripts/                    # Scripts for specific tasks (e.g., produce data by kafka etc)
├── logs/                       # different level of application log
├── config.py                   # Configuration file for global parameters (e.g., path, postgres cred)
├── main.py                     # Main script to execute end-to-end pipeline
├── requirements.txt            # Required libraries and dependencies
├── README.md                   # Project overview, setup, and instructions
└── .gitignore                  # Git ignore file for unnecessary files
```


## Getting Started

### Prerequisites
- Python 3.8 or higher

### Installation and Run Guide

#### 1. Clone the Repository (if applicable)

```bash
git clone git@github.com:hasnat-abdullah/metro_air_leak_anomaly_detection.git
cd metro_air_leak_anomaly_detection
```

#### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Put dataset

Put the dataset(.csv) at the directory`raw_data/`

#### 4. Configure the `src/config.py` file

Ensure paths and model parameters are correct in the configuration file.

#### 5. Run the Pipeline

```bash
python main.py
```


### Testing

To run tests, navigate to the project root directory and execute:

pytest tests/

### Logs

Logs are stored in the logs/ directory and provide detailed insights into each model’s training and evaluation process.

## Contributing

	1.	Fork the repository.
	2.	Create your feature branch (git checkout -b feature/your_feature).
	3.	Commit your changes (git commit -m 'Add some feature').
	4.	Push to the branch (git push origin feature/your_feature).
	5.	Open a pull request.

## License

This project is licensed under the MIT License - see the [MIT License](https://opensource.org/licenses/MIT) for details.