import os

# -------- Get the absolute path --------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# -------- POSTGRES CONFIG --------
DATABASE_URL = "postgresql+psycopg2://postgres:12345@localhost:5432/metro"
QUERY = """
SELECT 
    _timestamp, tp2, tp3, h1, dv_pressure, reservoirs, oil_temperature, 
    motor_current, comp, dv_eletric, towers, mpg, lps, pressure_switch, 
    oil_level, caudal_impulses, _status 
FROM air_compressor;
"""

# -------- CSV CONFIG --------
CSV_FILE_PATH = os.path.join(BASE_PATH, "raw_data/18B_Biofilter.csv")
TIME_COLUMN = "time"
VALUE_COLUMN = "Oxygen"
TARGET_COLUMN = "_status" # label column

# -------- DATA PLOT SAVE PATH --------
PLOT_SAVE_PATH = os.path.join(BASE_PATH, "plot_images")

# -------- TRAINED MODEL SAVE SAVE --------
TRAINED_MODEL_SAVE_PATH = os.path.join(BASE_PATH, "training_results")





