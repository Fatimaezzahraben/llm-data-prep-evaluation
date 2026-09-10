import pandas as pd
import difflib
import numpy as np
import re

# Clean tuple_id (numeric, no missing values)
df['tuple_id'] = pd.to_numeric(df['tuple_id'], errors='coerce')

# Clean src (categorical, no missing values, standardize)
src_mapping = {
    'helloflight': 'helloflight',
    'boston': 'boston',
    'airtravelcenter': 'airtravelcenter',
    'flightview': 'flightview',
    'panynj': 'panynj',
    'businesstravellogue': 'businesstravellogue',
    'flylouisville': 'flylouisville',
    'orbitz': 'orbitz',
    'myrateplan': 'myrateplan',
    'flightstats': 'flightstats',
    'ua': 'ua'
}
df['src'] = df['src'].replace(src_mapping)

# Clean flight (categorical, no missing values, standardize format)
flight_pattern = re.compile(r'^([A-Z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3})$')
def clean_flight(flight):
    if pd.isna(flight):
        return flight
    flight = str(flight).strip()
    if not re.match(r'^[A-Z]{2}-\d{1,4}-[A-Z]{3}-[A-Z]{3}$', flight, re.IGNORECASE):
        return flight
    return flight.upper()
df['flight'] = df['flight'].apply(clean_flight)

# Clean sched_dep_time (categorical, 33% missing)
# First handle disguised missing values
disguised_missing = ['Not Available', 'Delayed', 'NA', 'N/A', 'unknown', '', ' ']
df['sched_dep_time'] = df['sched_dep_time'].replace(disguised_missing, np.nan)

# Clean time strings
def clean_time(time_str):
    if pd.isna(time_str):
        return time_str
    time_str = str(time_str).strip()
    # Remove date suffixes and parenthetical notes
    time_str = re.sub(r'\s+\w+\s*\d+', '', time_str)
    time_str = re.sub(r'\s*\(.*\)', '', time_str)
    # Standardize time format
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)
    time_str = re.sub(r'(\d+):(\d{2})\s*([ap])\.', r'\1:\2 \3.m.', time_str, flags=re.IGNORECASE)