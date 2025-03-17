from feast import FeatureView, Field, BigQuerySource, Entity, ValueType, FeatureService
from feast.types import Float32, Int64, String, UnixTimestamp

# Assuming you have an entity for trips
trip_entity = Entity(
    name="unique_key",
    description="A taxi trip id",
    value_type=ValueType.STRING
)

# Define your data source (BigQuery) for trip-level details
trip_source = BigQuerySource(
    name="trip_source",
    table="carbon-relic-439014-t0.chicago_taxi.data",
    timestamp_field="timestamp"
)

trip_features = FeatureView(
    name="trip_features",
    entities=[trip_entity],
    schema=[
        Field(name="trip_miles", dtype=Float32),
        Field(name="payment_type", dtype=String),
        Field(name="trip_total", dtype=Float32),
        Field(name="company", dtype=String),
        Field(name="trip_start_timestamp", dtype=UnixTimestamp),
        Field(name="extras", dtype=Float32),
        Field(name="tolls", dtype=Float32),
        Field(name="tips", dtype=Float32),
        Field(name="pickup_latitude", dtype=Float32),
        Field(name="pickup_longitude", dtype=Float32),
        Field(name="pickup_community_area", dtype=Float32),
        Field(name="taxi_id", dtype=String),
    ],
    source=trip_source,
)


# Driver-level feature view (keyed on taxi_ID)
driver_entity = Entity(
    name="taxi_id",
    description="A taxi identifier",
    value_type=ValueType.STRING
)

driver_source = BigQuerySource(
    name="driver_source",
    table="carbon-relic-439014-t0.chicago_taxi.driver_aggregates",  # Assuming this table exists
    timestamp_field="timestamp"
)

driver_features = FeatureView(
    name="driver_features",
    entities=[driver_entity],
    schema=[
        Field(name="avg_tips", dtype=Float32),
    ],
    source=driver_source,
)

taxi_drive_fs = FeatureService(
    name="taxi_drive",
    features=[trip_features]
)
