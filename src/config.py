DATABASE_URL = "postgresql+psycopg2://postgres:12345@localhost:5432/metro"
QUERY = """
SELECT 
    _timestamp, tp2, tp3, h1, dv_pressure, reservoirs, oil_temperature, 
    motor_current, comp, dv_eletric, towers, mpg, lps, pressure_switch, 
    oil_level, caudal_impulses, _status 
FROM air_compressor;
"""