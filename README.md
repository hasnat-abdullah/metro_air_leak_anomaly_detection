
# Metro Air Leak Anomaly Detection Project

This project focuses on detecting anomalies in air compressor data to support predictive maintenance. It uses a variety of machine learning algorithms including Isolation Forest, Autoencoder, LSTM etc for detecting potential failures based on sensor data collected over several months [[**Dataset**](https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset)].

## Project Structure

metro_air_leak_anomaly_detection/
```
├── raw_data/                   # Raw data files (e.g., original CSV files)
├── src/                        # Source code for the project
│   ├── init.py                 # Init file for package recognition
│   ├── config.py               # Configuration file for global parameters (e.g., path, postgres cred)
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
│   └── utils.py                # helper functions
│   │   ├── {file_name}.py      # helper functions to reuse and keep the main functions clean
│   └── main.py                 # Main script to execute the pipeline end-to-end
├── tests/                      # Unit tests and integration tests
├── scripts/                    # Scripts for specific tasks (e.g., produce data by kafka etc)
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

This project is licensed under the MIT License - see the [MIT License](https://opensource.org/licenses/MIT) for details.