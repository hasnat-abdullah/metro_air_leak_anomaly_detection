from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'metro_air_compressor_data',
    bootstrap_servers='localhost:9093',
    group_id='metro-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    print(f"Received message: {message}")
    # To stop after 5 messages, uncomment the next line
    # if message.offset == 4:
    #     break