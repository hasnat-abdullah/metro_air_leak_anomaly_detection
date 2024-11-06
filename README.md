
# Metro Air Leak Anomaly Detection Project

This project focuses on detecting anomalies in air compressor data to support predictive maintenance. It uses a variety of machine learning algorithms including Isolation Forest, Autoencoder, LSTM etc for detecting potential failures based on sensor data collected over several months [[**Dataset**](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset)].

## Project Structure

metro_air_leak_anomaly_detection/
```
├── data/
│   ├── raw/                    # Raw data files (e.g., original CSV files)
│   ├── processed/              # Preprocessed data ready for model training
│
├── src/                        # Source code for the project
│   ├── init.py                 # Init file for package recognition
│   ├── data_preprocessing.py   # Data loading and preprocessing functions
│   ├── feature_engineering.py  # Feature engineering utilities
│   ├── models/                 # Folder for model classes
│   │   ├── init.py
│   │   ├── model_name.py       # specific model implementation
│   ├── evaluation.py           # Evaluation metrics and performance comparison functions
│   └── main.py                 # Main script to execute the pipeline end-to-end
│
├── config.yaml                 # Configuration file for global parameters (e.g., paths, model params)
├── tests/                      # Unit tests and integration tests
├── scripts/                    # Scripts for specific tasks (e.g., data processing, training)
├── logs/                       # different level of application log
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
git clone <your-repository-url>
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

Put the dataset(.csv) at the directory`data/raw/`

#### 4. Configure the `config/config.yaml` file

Ensure paths and model parameters are correct in the configuration file.

#### 5. Run the Pipeline

```bash
python src/main.py
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

Distributed under the MIT License. See LICENSE for more information.
