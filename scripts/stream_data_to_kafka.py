import time
import psycopg2
from kafka import KafkaProducer, KafkaConsumer
import json
import datetime
from abc import ABC, abstractmethod
import logging
from colorlog import ColoredFormatter

# Colorlog formatter
log_format = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s%(reset)s"
formatter = ColoredFormatter(log_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

#TODO: put credentials to env file
PG_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "metro",
    "user": "postgres",
    "password": "12345"
}
KAFKA_TOPIC = 'metro_air_compressor_data'
KAFKA_SERVERS = ['localhost:9093']
COLUMNS = ['_timestamp', 'tp2', 'tp3', 'h1', 'dv_pressure', 'reservoirs', 'oil_temperature', 'motor_current', 'comp',
           'dv_eletric', 'towers', 'mpg', 'lps', 'pressure_switch', 'oil_level', 'caudal_impulses', '_status']


class DataSource(ABC):
    """Abstract class for data source (e.g., PostgreSQL)."""

    @abstractmethod
    def fetch_data(self, last_fetched_timestamp):
        pass


class DataSink(ABC):
    """Abstract class for data sink (e.g., Kafka)."""

    @abstractmethod
    def send_data(self, data):
        pass


class PostgreSQLConnection(DataSource):
    """PostgreSQL database connection and data fetcher."""

    def __init__(self, config):
        self.config = config
        self.connection = self.connect()

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            logger.info("Connecting to PostgreSQL...")
            return psycopg2.connect(**self.config)
        except psycopg2.DatabaseError as e:
            logger.error(f"PostgreSQL connection error: {e}")
            raise

    def fetch_data(self, last_fetched_timestamp):
        """Fetch data from PostgreSQL."""
        query = """
        SELECT _timestamp, tp2, tp3, h1, dv_pressure, reservoirs, oil_temperature, motor_current, comp, dv_eletric, 
               towers, mpg, lps, pressure_switch, oil_level, caudal_impulses, _status
        FROM air_compressor 
        WHERE _timestamp > %s
        ORDER BY _timestamp ASC 
        LIMIT 100;
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (last_fetched_timestamp,))
            rows = cursor.fetchall()

        if rows:
            logger.info(f"Fetched {len(rows)} rows starting from {last_fetched_timestamp}")
            return rows, rows[-1][0]  # Update last fetched timestamp
        else:
            logger.info("No new data fetched.")
            return rows, last_fetched_timestamp


class KafkaDataProducer(DataSink):
    """Kafka producer to send data to Kafka topic."""

    def __init__(self, servers, topic):
        self.producer = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic

    def send_data(self, data):
        """Send data to Kafka."""
        try:
            self.producer.send(self.topic, value=data)
        except Exception as e:
            logger.error(f"Error sending data to Kafka: {e}")

    def close(self):
        """Close the Kafka producer."""
        self.producer.close()
        logger.info("Kafka producer closed.")


def convert_row_to_dict(columns, row):
    """Convert a row of data to a dictionary."""
    row_dict = dict(zip(columns, row))
    if isinstance(row_dict['_timestamp'], datetime.datetime):
        row_dict['_timestamp'] = row_dict['_timestamp'].strftime('%Y-%m-%d %H:%M:%S')

    return row_dict



class DataStreamer:
    """Main class to stream data from source to sink."""

    def __init__(self, data_source, data_sink, columns):
        self.data_source = data_source
        self.data_sink = data_sink
        self.columns = columns

    def start_streaming(self, initial_timestamp='2020-02-01 00:00:00'):
        last_fetched_timestamp = initial_timestamp
        try:
            logger.info("Starting data stream...")

            while True:
                rows, last_fetched_timestamp = self.data_source.fetch_data(last_fetched_timestamp)
                if rows:
                    for row in rows:
                        data = convert_row_to_dict(self.columns, row)
                        self.data_sink.send_data(data)
                logger.info(f"Last fetched timestamp: {last_fetched_timestamp}, sleeping for 30 seconds.")
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Data stream interrupted by user.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.data_sink.close()


def main():
    """Set up and start the data streaming process."""
    pg_connection = PostgreSQLConnection(PG_CONFIG)
    kafka_producer = KafkaDataProducer(KAFKA_SERVERS, KAFKA_TOPIC)

    data_streamer = DataStreamer(pg_connection, kafka_producer, COLUMNS)
    data_streamer.start_streaming()


if __name__ == "__main__":
    main()