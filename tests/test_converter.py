import json
from decimal import Decimal
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import boto3
import pytest
from moto import mock_aws
from schemaworks.utils import DecimalEncoder
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    ArrayType,
    DecimalType,
    MapType
)
from schemaworks.converter import JsonSchemaConverter
from botocore.exceptions import ClientError


@pytest.fixture
def json_schema_converter() -> Any:
    """
    Creates and returns an instance of JsonSchemaConverter.

    Returns:
        Any: An instance of JsonSchemaConverter.
    """
    return JsonSchemaConverter()


@pytest.fixture
def example_json_schema() -> Dict[str, Any]:
    """
    Creates and returns an example JSON schema.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema",
        "properties": {
            "uid": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
            "score": {"type": "float"}
        }
    }


@pytest.fixture
def example_full_json_schema() -> Dict[str, Any]:
    """
    Creates and returns an example JSON schema.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-full-schema",
        "properties": {
            "string_key": {"type": "string"},
            "number_key": {"type": "number"},
            "boolean_key": {"type": "boolean"},
            "integer_key": {"type": "integer"},
            "float_key": {"type": "float"},
            "date_key": {"type": "date"},
            "datetime_key": {"type": "datetime"},
            "long_key": {"type": "long"},
            "decimal_key": {"type": "decimal",
                            "properties": {
                                "precision": 5,
                                "scale": 2
                                }
                            },
            "array_key": {"type": "array",
                          "items": {"type": "string"}
                          },
            "Map_Key": {"type": "map",
                        "properties": {
                            "key": {"type": "integer"},
                            "value": {"type": "integer"}
                            }
                        }
        }
    }


@pytest.fixture
def example_json_schema_with_mapping() -> Dict[str, Any]:
    """
    Creates and returns an example JSON schema with nested object mapping.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema",
        "properties": {
            "uid": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
            "score": {"type": "float"},
            "nested": {
                "type": "object",
                "properties": {
                    "nested_uid": {"type": "string"},
                    "nested_age": {"type": "integer"}
                }
            }
        }
    }


@pytest.fixture
def example_mapping() -> Dict[str, Any]:
    """
    Creates and returns an example mapping.

    Returns:
        Dict[str, Any]: A dictionary representing the mapping.
    """
    return {
        "uid": {"user_id": {"type": "string"}},
        "nested": {
            "nested_data": {
                "type": "object",
                "properties": {
                    "nested_uid": {"type": "string"},
                    "nested_age": {"type": "integer"}
                }
            }
        }
    }


def test_load_schema_from_file(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Read JSON schema from file correctly."""
    with NamedTemporaryFile(delete=True, suffix=".json") as tmp:
        with open(tmp.name, "w", encoding="utf-8") as f:
            json.dump(example_json_schema, f)

        result = json_schema_converter.load_schema_from_file(tmp.name)
        assert result == example_json_schema
        assert json_schema_converter.json_schema == example_json_schema


@mock_aws
def test_load_schema_from_s3(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Read JSON schema from S3 without client or resource correctly."""
    s3 = boto3.client("s3", region_name="eu-west-1")
    bucket_name = "test-bucket"
    key = "schema.json"

    # Mock a bucket and an object
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={
                     "LocationConstraint": "eu-west-1"})
    s3.put_object(Bucket=bucket_name, Key=key, Body=json.dumps(
        example_json_schema).encode("utf-8"))

    s3_uri = f"s3://{bucket_name}/{key}"
    result = json_schema_converter.load_schema_from_s3(s3_uri)

    assert result == example_json_schema
    assert json_schema_converter.json_schema == example_json_schema


@mock_aws
def test_load_schema_from_s3_with_resource(json_schema_converter, example_json_schema):
    """Read JSON schema from S3 using provided session."""
    session = boto3.Session(region_name="eu-west-1")
    s3_resource = session.resource("s3", region_name="eu-west-1")
    bucket_name = "session-bucket"
    key = "schema.json"

    # create bucket and upload object using the explicit session
    s3_resource.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    s3_resource.Object(bucket_name, key).put(
        Body=json.dumps(example_json_schema).encode("utf-8")
    )

    s3_uri = f"s3://{bucket_name}/{key}"
    result = json_schema_converter.load_schema_from_s3(
        s3_uri,
        region="eu-west-1",
        client_or_resource=s3_resource
    )

    assert result == example_json_schema
    assert json_schema_converter.json_schema == example_json_schema


@mock_aws
def test_load_schema_from_s3_with_client(json_schema_converter, example_json_schema):
    """Read JSON schema from S3 using provided client."""
    s3 = boto3.client("s3", region_name="eu-west-1")
    bucket_name = "client-bucket"
    key = "schema.json"

    # create bucket and upload object using the explicit client
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(example_json_schema).encode("utf-8")
    )

    s3_uri = f"s3://{bucket_name}/{key}"
    result = json_schema_converter.load_schema_from_s3(
        s3_uri,
        region="eu-west-1",
        client_or_resource=s3
    )

    assert result == example_json_schema
    assert json_schema_converter.json_schema == example_json_schema


@mock_aws
def test_load_schema_from_s3_non_dict_json_raises(json_schema_converter):
    """Raise AttributeError when S3 returns non-dict JSON."""
    s3 = boto3.client("s3", region_name="eu-west-1")
    bucket_name = "test-bucket"
    key = "schema.json"

    # create bucket and upload a JSON array
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps([1, 2, 3]).encode("utf-8")
    )

    s3_uri = f"s3://{bucket_name}/{key}"
    with pytest.raises(AttributeError) as excinfo:
        json_schema_converter.load_schema_from_s3(s3_uri)
        assert "The loaded JSON is not a dictionary." in str(excinfo.value)


def test_load_schema_from_s3_invalid_client_or_resource(json_schema_converter):
    """Raise ValueError when client_or_resource is not a boto3 client or resource."""
    with pytest.raises(TypeError) as excinfo:
        json_schema_converter.load_schema_from_s3("s3://bucket/key", client_or_resource="invalid")
        assert "client_or_resource must be a boto3 client or resource." in str(excinfo.value)


@mock_aws
def test_load_schema_from_s3_invalid_schema_path(json_schema_converter):
    """Raise ValueError for invalid S3 URI."""
    with pytest.raises(ValueError) as excinfo:
        json_schema_converter.load_schema_from_s3("any-bucket/any-key")
        assert "Invalid S3 URI: any-bucket/any-key" in str(excinfo.value)


@mock_aws
def test_load_schema_from_s3_client_error_prints_and_raises(json_schema_converter, capsys):
    """print error and re-raise clienterror when s3 get_object fails for client."""
    s3_client = boto3.client("s3", region_name="eu-west-1")
    # monkeypatch get_object to raise ClientError
    error_response = {'Error': {'Code': 'NoSuchKey', 'Message': 'Key not found'}}
    def fake_get_object(**kwargs):
        raise ClientError(error_response, 'GetObject')
    s3_client.get_object = fake_get_object

    s3_uri = "s3://bucket/key"
    with pytest.raises(ClientError):
        json_schema_converter.load_schema_from_s3(s3_uri, client_or_resource=s3_client)
        captured = capsys.readouterr()
        assert "An error ocurred when reading schema from S3." in captured.out


def test_apply_mapping(json_schema_converter: Any, example_mapping: Dict[str, Any]) -> None:
    """Apply mapping correctly to converter mapping attribute."""
    json_schema_converter.apply_mapping(example_mapping)
    assert json_schema_converter.mapping == example_mapping


def test_to_spark_schema(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Convert JSON schema to Spark StructType correctly."""
    json_schema_converter.json_schema = example_json_schema
    spark_schema = json_schema_converter.to_spark_schema()

    expected_schema = StructType([
        StructField("uid", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("score", FloatType(), True)
    ])

    assert spark_schema == expected_schema


def test_to_spark_schema_no_json_schema_raises(json_schema_converter):
    """Raise AttributeError when no JSON schema is set for Spark conversion."""
    converter = json_schema_converter
    # json_schema fixture initializes to empty dict
    with pytest.raises(AttributeError) as excinfo:
        converter.to_spark_schema()
    assert "No JSON schema available" in str(excinfo.value)


def test_to_sql_string_no_json_schema_raises(json_schema_converter):
    """Raise AttributeError when no JSON schema is set for SQL string."""
    converter = json_schema_converter
    with pytest.raises(AttributeError) as excinfo:
        converter.to_sql_string()
    assert "No JSON schema available" in str(excinfo.value)


def test_to_spark_string(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Convert JSON schema to Spark string correctly."""
    json_schema_converter.json_schema = example_json_schema
    _ = json_schema_converter.to_spark_schema()
    spark_string = json_schema_converter.to_spark_string()

    expected_string = (
        'StructType([\n'
        '    StructField("uid", StringType(), nullable=True),\n'
        '    StructField("age", IntegerType(), nullable=True),\n'
        '    StructField("is_active", BooleanType(), nullable=True),\n'
        '    StructField("score", FloatType(), nullable=True),\n'
        '])'
    )
    assert spark_string == expected_string


def test_to_sql_string(json_schema_converter: Any, example_json_schema: Any, example_full_json_schema: Any) -> None:
    """Convert JSON schema to SQL string correctly."""
    json_schema_converter.json_schema = example_json_schema
    sql_string = json_schema_converter.to_sql_string()

    expected_sql_string = "uid string, age int, is_active boolean, score float"
    assert sql_string == expected_sql_string

    json_schema_converter.json_schema = example_full_json_schema
    full_sql_string = json_schema_converter.to_sql_string(to_lower=True)
    assert full_sql_string == "string_key string, " + \
        "number_key float, " + \
        "boolean_key boolean, " + \
        "integer_key int, " + \
        "float_key float, " + \
        "date_key date, " + \
        "datetime_key timestamp, " + \
        "long_key bigint, " + \
        "decimal_key decimal(5, 2), " + \
        "array_key array<string>, " + \
        "map_key map<int, int>"


def test_to_dtypes(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Convert JSON schema to dictionary of dtypes correctly."""
    json_schema_converter.json_schema = example_json_schema
    dtypes = json_schema_converter.to_dtypes()

    expected_dtypes: Dict[str, str] = {
        "uid": "string",
        "age": "int",
        "is_active": "boolean",
        "score": "float"
    }
    assert dtypes == expected_dtypes


def test_to_dtypes_no_json_schema_raises(json_schema_converter):
    """Raise AttributeError when no JSON schema is set for dtypes."""
    converter = json_schema_converter
    with pytest.raises(AttributeError) as excinfo:
        converter.to_dtypes()
    assert "No JSON schema available" in str(excinfo.value)


def test_to_dtypes_non_string_val_raises(json_schema_converter, monkeypatch):
    """Raise AttributeError if _to_sql_string returns non-string in dtypes."""
    json_schema_converter.json_schema = {"properties": {"x": {"type": "string"}}}
    monkeypatch.setattr(json_schema_converter, '_to_sql_string', lambda data, to_lower=False: 123)
    with pytest.raises(AttributeError) as excinfo:
        json_schema_converter.to_dtypes()
    assert "'_to_sql_string' returned a non-string value." in str(excinfo.value)


def test_to_dtypes_lowercase_keys(json_schema_converter):
    """Lowercase keys in dtypes output when to_lower is True."""
    json_schema_converter.json_schema = {
        "properties": {
            "KeyOne": {"type": "string"},
            "KeyTwo": {"type": "integer"}
        }
    }
    result = json_schema_converter.to_dtypes(to_lower=True)
    assert "keyone" in result and result["keyone"] == "string"
    assert "keytwo" in result and result["keytwo"] == "int"


def test_decimal_encoder() -> None:
    """Encode and decode Decimal and float values using DecimalEncoder."""
    data: Dict[str, Any] = {
        "decimal_value": Decimal("12.34"),
        "integer_value": Decimal("12"),
        "float_value": 12.34
    }

    encoded = json.dumps(data, cls=DecimalEncoder)
    decoded = json.loads(encoded)

    assert decoded["decimal_value"] == 12.34
    assert decoded["integer_value"] == 12
    assert decoded["float_value"] == 12.34


def test_flatten_schema_basic() -> None:
    """Flatten schema with nested properties using default separator."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema",
        "properties": {
            "uid": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "nested1": {"type": "number"},
                    "nested2": {"type": "string"}
                }
            }
        }
    }
    expected_output = {
        "uid": "string",
        "details.nested1": "number",
        "details.nested2": "string"
    }
    assert JsonSchemaConverter._flatten_schema(schema) == expected_output


def test_flatten_schema_with_different_separator() -> None:
    """Flatten schema using a different separator."""
    schema = {
        "properties": {
            "uid": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "nested1": {"type": "number"},
                    "nested2": {"type": "string"}
                }
            }
        }
    }
    expected_output = {
        "uid": "string",
        "details/nested1": "number",
        "details/nested2": "string"
    }
    assert JsonSchemaConverter._flatten_schema(schema, sep="/") == expected_output


def test_flatten_schema_with_no_properties() -> None:
    """Return empty dict when flattening schema with no properties."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema"
    }
    expected_output: dict[str, Any] = {}
    assert JsonSchemaConverter._flatten_schema(schema) == expected_output


def test_to_flat(json_schema_converter: JsonSchemaConverter) -> None:
    """Convert JSON schema to flat dictionary using default separator."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema",
        "properties": {
            "uid": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "nested1": {"type": "number"},
                    "nested2": {"type": "string"}
                }
            }
        }
    }
    json_schema_converter.json_schema = schema
    expected_output = {
        "uid": "string",
        "details.nested1": "number",
        "details.nested2": "string"
    }
    assert json_schema_converter.to_flat() == expected_output


def test_to_flat_with_different_separator(json_schema_converter: JsonSchemaConverter) -> None:
    """Convert JSON schema to flat dictionary with custom separator."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema",
        "properties": {
            "uid": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "nested1": {"type": "number"},
                    "nested2": {"type": "string"}
                }
            }
        }
    }
    json_schema_converter.json_schema = schema
    expected_output = {
        "uid": "string",
        "details/nested1": "number",
        "details/nested2": "string"
    }
    assert json_schema_converter.to_flat(sep="/") == expected_output


def test_to_spark_string_initializes_schema(json_schema_converter: Any, example_json_schema: Any) -> None:
    """Initialize Spark schema and return Spark string when Spark schema is None."""
    # Ensure spark_schema is initially None
    assert json_schema_converter.spark_schema is None

    # Set json_schema and call to_spark_string without prior to_spark_schema
    json_schema_converter.json_schema = example_json_schema
    spark_string = json_schema_converter.to_spark_string()

    # Spark_schema should now be set
    assert json_schema_converter.spark_schema is not None

    # Verify the returned string is correct
    expected_string = (
        'StructType([\n'
        '    StructField("uid", StringType(), nullable=True),\n'
        '    StructField("age", IntegerType(), nullable=True),\n'
        '    StructField("is_active", BooleanType(), nullable=True),\n'
        '    StructField("score", FloatType(), nullable=True),\n'
        '])'
    )
    assert spark_string == expected_string


def test_to_spark_schema_object_type(json_schema_converter):
    """Convert JSON schema with object root type to Spark StructType."""
    json_schema_converter.json_schema = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"}
        }
    }
    spark_schema = json_schema_converter.to_spark_schema()
    expected = StructType([
        StructField("field1", StringType(), True),
        StructField("field2", IntegerType(), True)
    ])
    assert spark_schema == expected


@pytest.mark.parametrize(
    "schema, expected_class, check",
    [
        (  # array type
            {"type": "array", "items": {"type": "string"}},
            ArrayType,
            lambda s: isinstance(s.elementType, StringType),
        ),
        (  # decimal type
            {"type": "decimal", "properties": {"precision": 12, "scale": 3}},
            DecimalType,
            lambda s: (s.precision == 12 and s.scale == 3),
        ),
        (  # map type
            {"type": "map", "properties": {"key": {"type": "string"}, "value": {"type": "integer"}}},
            MapType,
            lambda s: isinstance(s.keyType, StringType) and isinstance(s.valueType, IntegerType),
        ),
    ],
)
def test_to_spark_schema_varieties(json_schema_converter, schema, expected_class, check):
    """Convert array, decimal, and map types in JSON schema to Spark types."""
    json_schema_converter.json_schema = schema
    result = json_schema_converter.to_spark_schema()
    assert isinstance(result, expected_class)
    assert check(result)


def test_to_spark_schema_with_mapping_root(json_schema_converter):
    """Apply mapping and convert root properties to Spark schema."""
    json_schema_converter.json_schema = {
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "integer"}
        }
    }
    # Map 'a' to 'alpha' with type float
    json_schema_converter.apply_mapping({
        "a": {"alpha": {"type": "float"}}
    })
    spark_schema = json_schema_converter.to_spark_schema()
    # Expected: 'alpha' as FloatType and 'b' unchanged as IntegerType
    expected = StructType([
        StructField("alpha", FloatType(), True),
        StructField("b", IntegerType(), True)
    ])
    assert spark_schema == expected


def test_to_spark_schema_unknown_type_raises(json_schema_converter):
    """Raise AttributeError for unknown data type in Spark schema."""
    json_schema_converter.json_schema = {"type": "unknown"}
    with pytest.raises(AttributeError) as excinfo:
        json_schema_converter.to_spark_schema()
    assert "Unknown data type" in str(excinfo.value)


@pytest.mark.parametrize(
    "schema, expected",
    [
        (
            {"type": "array", "items": {"type": "integer"}},
            "ArrayType(IntegerType(), containsNull=True)",
        ),
        (
            {"type": "map", "properties": {"key": {"type": "string"}, "value": {"type": "boolean"}}},
            "MapType(StringType(), BooleanType(), valueContainsNull=True)",
        ),
        (
            {"type": "decimal", "properties": {"precision": 8, "scale": 2}},
            "DecimalType(8,2)",
        ),
    ],
)
def test_to_spark_string_varieties(json_schema_converter, schema, expected):
    """Output correct Spark string for array, map, and decimal root types."""
    json_schema_converter.json_schema = schema
    json_schema_converter.to_spark_schema()
    output = json_schema_converter.to_spark_string()
    assert output == expected


def test_to_spark_string_unknown_datatype_raises(json_schema_converter):
    """Raise AttributeError for unknown datatype in spark string."""
    with pytest.raises(AttributeError) as excinfo:
        # BinaryType is not handled explicitly in _to_spark_string
        json_schema_converter._to_spark_string(BinaryType())
    assert "Unknown data type" in str(excinfo.value)


def test_to_sql_string_null_type_raises(json_schema_converter):
    """Raise ValueError for null type in SQL string."""
    json_schema_converter.json_schema = {"properties": {"x": {"type": "null"}}}
    with pytest.raises(ValueError) as excinfo:
        json_schema_converter.to_sql_string()
    assert "not supported" in str(excinfo.value)


def test_to_sql_string_nested_object(json_schema_converter):
    """Convert nested object in JSON schema to SQL string."""
    json_schema_converter.json_schema = {
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "integer"}
                }
            }
        }
    }
    sql_string = json_schema_converter.to_sql_string(to_lower=True)
    assert sql_string == "outer struct<inner: int>"


def test_to_sql_string_with_mapping(json_schema_converter):
    """Apply mapping and convert to SQL string correctly."""
    json_schema_converter.json_schema = {"properties": {"cnt": {"type": "integer"}}}
    json_schema_converter.apply_mapping({"cnt": {"count": {"type": "long"}}})
    sql_string = json_schema_converter.to_sql_string()
    assert sql_string == "count bigint"


def test_to_sql_string_unknown_data_type_raises(json_schema_converter):
    """Raise AttributeError for unknown data type in SQL conversion."""
    with pytest.raises(AttributeError) as excinfo:
        json_schema_converter._to_sql_string({"type": "foo"}, False)
    assert "Unknown data type 'foo'." in str(excinfo.value)


def test_to_sql_string_non_string_output_raises(json_schema_converter, monkeypatch):
    """Raise ValueError if _to_sql_string returns non-string in SQL string."""
    json_schema_converter.json_schema = {"properties": {"x": {"type": "string"}}}
    # monkeypatch _to_sql_string to return a non-string type
    monkeypatch.setattr(json_schema_converter, '_to_sql_string', lambda data, to_lower=False: 123)
    with pytest.raises(ValueError) as excinfo:
        json_schema_converter.to_sql_string()
    assert "The output of 'to_sql_string' is not a string." in str(excinfo.value)


@pytest.mark.parametrize(
    "schema, expected",
    [
        (
            {"type": "array", "items": {"type": "integer"}},
            "array<int>",
        ),
        (
            {"type": "map", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}},
            "map<string, string>",
        ),
    ],
)
def test_to_sql_string_array_and_map_varieties(json_schema_converter, schema, expected):
    """Output correct SQL string for array and map root types."""
    json_schema_converter.json_schema = schema
    result = json_schema_converter.to_sql_string()
    assert result == expected
