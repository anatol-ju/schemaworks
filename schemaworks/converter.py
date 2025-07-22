import json
import logging
from typing import Any
from urllib.parse import urlparse

import boto3
import botocore
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
from pyiceberg.schema import Schema as IcebergSchema
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType
)

from schemaworks.utils import ATHENA_TYPE_MAP, SPARK_TYPE_MAP, IcebergIDAllocator

LOGGER = logging.getLogger(__name__)

class JsonSchemaConverter():
    """
    This class provides convenience functions to read JSON schema files from
    local or S3 and convert them into PySpark schema objects or string representations
    for Spark or SQL.
    The data types are extended to match databases, allowing specifying types such as:
        - long
        - date
        - datetime
        - timestamp

    These are converted appropriate for the platform (Spark or AWS Athena).
    """
    def __init__(self, schema: dict[str, Any] = {}) -> None:
        """
        Initializes the converter with an optional JSON schema.

        Args:
            schema (dict[str, Any]): A JSON schema in dictionary format.
                If not provided, an empty schema is initialized.
        """
        self.json_schema: dict[str, Any] = schema
        self.spark_schema: DataType|None = None
        self.spark_string: str = ""
        self.mapping: dict[str, dict[str, Any]] = {}
        self.sql_schema_string: str = ""

    def _to_spark_schema(self, data: dict[str, Any], to_lower: bool) -> DataType:
        """
        Helper function to parse the dictionary recursively and therefore
        resolve all nested structures.

        Args:
            data (dict[str, Any]): Either the initial dataset, the sub-dataset
                used for recursion, or a string to be resolved to a data type.
            to_lower (bool): Convert all field names to lower case.

        Raises:
            AttributeError: Raised if any of the fields have data type `null`.
            AttributeError: Raised if the data type can not be resolved.

        Returns:
            DataType: The sub-dataset used for recursion, or a PySpark DataType object.
        """
        if "type" not in data or not isinstance(data["type"], str):
            if "properties" in data:
                return self._to_spark_schema(data["properties"], to_lower)
            else:
                r = StructType()
                for key, value in data.items():
                    if key in self.mapping:
                        k = list(self.mapping[key].keys())[0]
                        v = list(self.mapping[key].values())[0]
                        r.add(StructField(k, self._to_spark_schema(v, to_lower)))
                    else:
                        r.add(StructField(key.lower() if to_lower else key, self._to_spark_schema(value, to_lower)))
                return r
        elif data["type"] == "array":
            return ArrayType(self._to_spark_schema(data["items"], to_lower))
        elif data["type"] == "object":
            return self._to_spark_schema(data["properties"], to_lower)
        elif data["type"] == "decimal":
            return DecimalType(int(data["properties"]["precision"]), int(data["properties"]["scale"]))
        elif data["type"] == "map":
            return MapType(self._to_spark_schema(data["properties"]["key"], to_lower), self._to_spark_schema(data["properties"]["value"], to_lower))
        else:
            data_type = SPARK_TYPE_MAP.get(data["type"], None)
            if data_type:
                return data_type
            else:
                LOGGER.error("Unknown data type %s.", data["type"])
                raise AttributeError(f"Unknown data type {data['type']}.")

    def _to_spark_string(self, field: DataType, depth: int = 1) -> str:
        """
        Helper function to convert a PySpark DataType object to a string representation.

        Args:
            field (DataType): The PySpark DataType object to convert.
            depth (int): The current depth in the schema, used for formatting.

        Raises:
            AttributeError: Raised if the provided field is not a recognized data type.

        Returns:
            str: A string representation of the PySpark DataType object.
        """
        if isinstance(field, (NullType, StringType, BooleanType, IntegerType, FloatType, DateType, TimestampNTZType, LongType)):
            return str(field)
        elif isinstance(field, ArrayType):
            r = "ArrayType("
            r += self._to_spark_string(field.elementType, depth)
            r += f", containsNull={field.containsNull})"
            return r
        elif isinstance(field, DecimalType):
            r = f"DecimalType({field.precision},{field.scale})"
            return r
        elif isinstance(field, StructType):
            r = "StructType(["
            r += "\n" if len(field)>0 else ""
            for sf in field.fields:
                r += "    "*depth if len(field)>0 else ""
                r += self._to_spark_string(sf, depth + 1) + "\n"
            r += "    " * (depth - 1) + "])"
            return r
        elif isinstance(field, MapType):
            r = "MapType("
            r += self._to_spark_string(field.keyType, depth)
            r += ", "
            r += self._to_spark_string(field.valueType, depth)
            r += f", valueContainsNull={str(field.valueContainsNull)})"
            return r
        elif isinstance(field, StructField):
            r = f"StructField(\"{field.name}\", "
            r += self._to_spark_string(field.dataType, depth)
            r += f", nullable={field.nullable}),"
            return r
        else:
            LOGGER.error("Unknown data type %s.", field)
            raise AttributeError(f"Unknown data type {field}.")

    def _to_sql_string(self, data: dict[str, Any], to_lower: bool = False) -> str|dict[str, Any]:
        """
        Helper function to parse the dictionary recursively and therefore
        resolve all nested structures.

        Args:
            data (dict[str, Any]): Either the initial dataset, the sub-dataset
                used for recursion, or a string to be resolved to a data type.

        Raises:
            ValueError: Raised if any of the fields have data type `null`.
            ValueError: Raised if the data type can not be resolved.

        Returns:
            str|dict[str, Any]: The sub-dataset used for recursion,
                or a string representing a data type.
        """
        if "type" not in data or not isinstance(data["type"], str):
            if "properties" in data:
                return self._to_sql_string(data["properties"], to_lower)
            else:
                arr = []
                for key in data.keys():
                    if self.mapping and key in self.mapping:
                        new_key, new_value = next(iter(self.mapping[key].items()))
                        arr.append(f"{new_key.lower() if to_lower else new_key} {self._to_sql_string(new_value)}")
                    else:
                        arr.append(f"{key.lower() if to_lower else key} {self._to_sql_string(data[key])}")
                return f"struct<{', '.join(arr)}>"
        elif data["type"] == "null":
            LOGGER.error("Data type 'null' is not supported by Iceberg schema.")
            raise ValueError(
                f"Data type '{data['type']}' is not supported by Iceberg schema."
            )
        elif data["type"] == "array":
            return f"array<{self._to_sql_string(data['items'])}>"
        elif data["type"] == "object":
            arr = []
            for key in data["properties"].keys():
                arr.append(f"{key.lower() if to_lower else key}: {self._to_sql_string(data['properties'][key])}")
            return f"struct<{', '.join(arr)}>"
        elif data["type"] == "decimal":
            return f"decimal({data['properties']['precision']}, {data['properties']['scale']})"
        elif data["type"] == "map":
            return f"map<{self._to_sql_string(data['properties']['key'])}, {self._to_sql_string(data['properties']['value'])}>"
        else:
            data_type = ATHENA_TYPE_MAP.get(data["type"], None)
            if data_type:
                return data_type
            else:
                LOGGER.error("Unknown data type '%s'.", data["type"])
                raise AttributeError(f"Unknown data type '{data['type']}'.")

    def load_schema_from_file(self, filepath: str) -> dict[str, Any]:
        """
        Read a local file formatted as JSON schema and returns the Python object.

        Args:
            filepath (str): The full path to the local JSON file.
        """
        with open(filepath, encoding="utf-8") as reader:
            self.json_schema = json.load(reader)
        return self.json_schema

    def load_schema_from_s3(
        self,
        s3_uri: str,
        region: str = "eu-west-1",
        client_or_resource: Any | None = None,
    ) -> dict[str, Any]:
        """
        Reads a JSON file from S3 and provides it's contents in form of a dictionary.

        Args:
            s3_uri (str): The path to the JSON file, i.e. `s3://bucket/key.json`.
            region (str): The AWS region name. Defaults to "eu-west-1".
            client_or_resource (Any | None): A boto3 S3 resource or client.
                If not provided, a new resource is created using the specified region.

        Raises:
            ValueError: Raised when the provided path could not be resolved.

        Returns:
            dict[str, Any]: A dictionary representation of the JSON file.
        """
        s3_resource = None
        s3_client = None
        if not client_or_resource:
            s3_resource = boto3.resource("s3", region_name=region)
        elif isinstance(client_or_resource, ServiceResource):
            s3_resource = client_or_resource
        elif isinstance(client_or_resource, BaseClient):
            s3_client = client_or_resource
        else:
            LOGGER.error("Invalid client_or_resource parameter type: %s", type(client_or_resource))
            raise TypeError("client_or_resource parameter value must be a boto3 S3 resource or client.")

        schema_path = urlparse(s3_uri, allow_fragments=False)

        # Ensure urlparse returned a valid result with netloc and path
        if not hasattr(schema_path, "netloc") or not hasattr(schema_path, "path") \
            or not schema_path.netloc or not schema_path.path:
            LOGGER.error("Invalid S3 URI provided: %s", s3_uri)
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        try:
            if s3_resource:
                obj = s3_resource.Object(
                    schema_path.netloc, schema_path.path.split("/", 1)[-1]
                )
                content = obj.get()["Body"].read().decode("utf-8")
            elif s3_client:
                response = s3_client.get_object(
                    Bucket=schema_path.netloc, Key=schema_path.path.split("/", 1)[-1]
                )
                content = response["Body"].read().decode("utf-8")

            result = json.loads(content)

            if isinstance(result, dict):
                self.json_schema = result
                return result
            else:
                LOGGER.error("The loaded JSON is not a dictionary.")
                raise AttributeError("The loaded JSON is not a dictionary.")
        except botocore.exceptions.ClientError as err:
            LOGGER.error("An error ocurred when reading schema from S3.", exc_info=err)
            raise

    def apply_mapping(self, conversion_map: dict[str, dict[str, Any]] = {}) -> None:
        """
        Apply mapping to columns, enabling transforms of column names and types.
        The provided mapping is used in all other functions, so that all properties except
        `json_schema` contain the mapped data types.

        The conversion map entry must have the current column name as key and a dictionary object
        as a value, that has the exact same format as the JSON schema (see example below).

        This can be used to extend the capabilities of the JSON schema, that can't handle
        certain data types that are frequent in databases.

        #### IMPORTANT
            Any "custom" data types used in the mapping that are not part of the original
            JSON schema specification, must be handled in the functions. Otherwise an
            `AttributeError` is raised.

        Currently supported types are:
            - `float` signed 32-bit floating point numbers
            - `date`: objects of type `datetime.date`
            - `datetime`: objects of type `datetime.datetime` without timezone,
            handled as `TimestampNTZType` by PySpark
            - `timestamp`: unix epoch time in microseconds, handled as `LongType` by PySpark
            - `long`: signed 64-bit integer
            - `decimal`: decimal objects are numbers with precision (digits before the `.`)
                        and scale (digits after the `.`), use like
                        ```"properties": {"precision": 5, "scale": 2}```
            - `map`: dictionary without column name information, use like
                    ```"properties": {
                            "key": {"type": "string"},
                            "value": {"type": "integer"}
                    }```

        #### Example 1:
            A column defined as
            ```
            "ts": {"type": "number"}
            ```
            with a mapping
            ```
            mapping = {"ts": {"unix_ts": {"type": "long"}}}
            ```
            is renamed to `unix_ts` with a `long` type depending on the implementation (PySpark or SQL).

        #### Example 2:
            A column
            ```
            "json_str": {"type": "string"}
            ```
            with mapping
            ```
            mapping = {
                "json_str": {
                    "json_map": {
                        "type": "map",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "integer"}
                        }
                    }
                }
            }
            ```
            is renamed to `json_map` and uses key-value pairs instead of a string.
        """
        self.mapping = conversion_map

    def to_spark_schema(self, to_lower: bool = False) -> DataType:
        """
        Creates a schema for Spark DataFrames from a JSON schema.

        Args:
            to_lower (bool): Convert all field names to lowercase.
                This does not apply to mappings, if available.

        Raises:
            AttributeError: Raised if no schema data is available to convert.

        Returns:
            DataType: A PySpark DataType object as the schema.
        """
        if not self.json_schema:
            log_message = "No JSON schema available. Use 'read_json' or set the 'json_schema' attribute."
            LOGGER.error(log_message)
            raise AttributeError(log_message)
        self.spark_schema = self._to_spark_schema(self.json_schema, to_lower)
        return self.spark_schema

    def to_spark_string(self) -> str:
        """
        Creates a string representation of a schema object for a Spark DataFrame.
        This can be copied into the code to maually adjust data types, because
        JSON schemas don't support many table specific data types like `timestamp`.

        To make use of this tool, `print` it's output and copy and paste the result.
        Without this, the output is only a raw unformatted string.
        """
        if not self.spark_schema:
            self.spark_schema = self.to_spark_schema()
        self.spark_string = self._to_spark_string(self.spark_schema)
        return self.spark_string

    def to_sql_string(self, to_lower: bool = False) -> str:
        """
        Converts the schema data to a SQL schema string.
        If mapping data is available, it is applied automatically.
        Run `read_mapping` method to read mappings data from S3.

        All column names are converted to lower case, because the table saves all
        columns in lower case too.

        Example:
            ```
            data = {"ts": {"type": "number"}, "id": {"type": "string"}}
            ```
            results in
            ```
            "ts float, id string"
            ```

        Raises:
            ValueError: Raised if the result is not a valid string.

        Returns:
            str: The string representation of the schema as used by SQL DDL.
        """
        if not self.json_schema:
            log_message = "No JSON schema available. Use 'read_json' or set the 'json_schema' attribute."
            LOGGER.error(log_message)
            raise AttributeError(log_message)

        schema_str = self._to_sql_string(self.json_schema, to_lower)
        if isinstance(schema_str, str):
            self.sql_schema_string = schema_str
        else:
            LOGGER.error("'to_sql_string' returned a non-string value: %s", type(schema_str))
            raise ValueError("The output of 'to_sql_string' is not a string.")

        if self.sql_schema_string.startswith("struct<") and self.sql_schema_string.endswith(">"):
            self.sql_schema_string = (
                f"{self.sql_schema_string.split('struct<', 1)[-1].rsplit('>', 1)[0]}"
            )
        return self.sql_schema_string

    def to_dtypes(self, to_lower: bool = False) -> dict[str, str]:
        """
        Similar to `to_string`, but creates a dictionary from the schema.
        The keys are the column names and the values the SQL schemas.

        Args:
            to_lower (bool): If `True` all keys are converted to lower case.
                Defaults to `False`.

        Example:
            ```json
            {
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
            ```
            resolves to
            ```python
            {"uid": "string", "details": "struct<nested1: float, nested2: string>"}
            ```
        Returns:
            dict[str, str]: A dictionary with column names as keys and SQL schemas as their values.
        """
        if not self.json_schema:
            log_message = "No JSON schema available. Use 'read_json' or set the 'json_schema' attribute."
            LOGGER.error(log_message)
            raise AttributeError(log_message)

        result: dict[str, str] = {}
        for key, value in self.json_schema["properties"].items():
            val = self._to_sql_string(value, to_lower)
            if isinstance(val, str):
                result[key.lower() if to_lower else key] = val
            else:
                LOGGER.error("'_to_sql_string' returned a non-string value: %s", type(val))
                raise AttributeError("'_to_sql_string' returned a non-string value.")
        return result

    def to_flat(self, sep: str = ".") -> dict[str, str]:
        """
        Flattens the provided JSON schema.
        - Removes keywords like `type` and `properties`.
        - Handles nested fields by separating them by `.`.

        Args:
            sep (str, optional): Separator to use when unnesting fields. Defaults to ".".

        Returns:
            dict[str, str]: A flattened dictionary.

        Example:
            ```json
            {
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
            ```
            resolves to
            ```python
            {"uid": "string", "details.nested1": "float", "details.nested2": "string"}
            ```
        """
        from schemaworks.utils import flatten_schema
        return flatten_schema(self.json_schema, sep=sep)

    def to_iceberg_schema(self, id_start: int = 1) -> IcebergSchema:
        """
        Converts the JSON schema to an Iceberg schema by parsing the existing JSON.

        Args:
            id_start (int): The starting ID for the Iceberg schema fields. Defaults to
                1, which is a common starting point for Iceberg schemas.
        """
        if not self.json_schema:
            log_message = "No JSON schema available. Use 'read_json' or set the 'json_schema' attribute."
            LOGGER.error(log_message)
            raise AttributeError(log_message)

        from schemaworks.utils import build_iceberg_struct_type

        allocator = IcebergIDAllocator(id_start)
        required_fields = self.json_schema.get("required", [])
        properties = self.json_schema.get("properties", {})

        if not properties:
            log_message = "No properties found in the JSON schema."
            LOGGER.error(log_message)
            raise ValueError(log_message)

        struct_type = build_iceberg_struct_type(properties, allocator, required_fields)
        return IcebergSchema(*struct_type.fields)
