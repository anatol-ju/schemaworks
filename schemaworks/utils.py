import json
import logging
from decimal import Decimal
from typing import Any

from pyspark.sql.types import (
    BooleanType,
    DateType,
    FloatType,
    IntegerType,
    LongType,
    NullType,
    StringType,
    TimestampNTZType,
)

from pyiceberg.types import (
    BooleanType as IcebergBooleanType,
    DateType as IcebergDateType,
    FloatType as IcebergFloatType,
    IntegerType as IcebergIntegerType,
    LongType as IcebergLongType,
    DoubleType as IcebergDoubleType,
    StringType as IcebergStringType,
    TimestampType as IcebergTimestampType,
    DecimalType as IcebergDecimalType,
    MapType as IcebergMapType,
    ListType as IcebergListType,
    StructType as IcebergStructType,
    NestedField as IcebergNestedField
)

LOGGER = logging.getLogger(__name__)

ATHENA_TYPE_MAP = {
    "string": "string",
    "str": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "number": "float",
    "integer": "int",
    "int": "int",
    "long": "bigint",
    "float": "float",
    "date": "date",
    "datetime": "timestamp",
    "timestamp": "bigint"
}

SPARK_TYPE_MAP = {
    "null": NullType(),
    "string": StringType(),
    "str": StringType(),
    "boolean": BooleanType(),
    "bool": BooleanType(),
    "number": FloatType(),
    "integer": IntegerType(),
    "int": IntegerType(),
    "long": LongType(),
    "float": FloatType(),
    "date": DateType(),
    "datetime": TimestampNTZType(),
    "timestamp": LongType()
}

ICEBERG_TYPE_MAP = {
    "string": IcebergStringType,
    "str": IcebergStringType,
    "boolean": IcebergBooleanType,
    "bool": IcebergBooleanType,
    "number": IcebergFloatType,
    "integer": IcebergIntegerType,
    "int": IcebergIntegerType,
    "long": IcebergLongType,
    "float": IcebergFloatType,
    "double": IcebergDoubleType,
    "decimal": IcebergDecimalType,
    "date": IcebergDateType,
    "datetime": IcebergTimestampType,
    "timestamp": IcebergTimestampType
}


class DecimalEncoder(json.JSONEncoder):
    """
    Custom encoder to replace any `Decimal` objects with `int` or `float`.
    Also converts `float` to `int` if possible.
    """
    def default(self, obj: Any) -> Any:
        # Convert Decimal objects
        if isinstance(obj, Decimal):
            num = float(obj)
            # To int or float
            if num.is_integer():
                return int(obj)
            else:
                return float(obj)
        else:
            # Otherwise use the default behavior
            return json.JSONEncoder.default(self, obj)


class IcebergIDAllocator:
    """
    A simple ID allocator that generates unique IDs starting from a given number.
    The IDs are generated sequentially, starting from the specified `start` value.

    This class is useful for generating unique identifiers for Iceberg tables or other
    data structures where IDs are required that are globally consistent and safe.

    Usage:
    ```python
    from schemaworks.utils import IcebergIDAllocator

    allocator = IcebergIDAllocator(start=1000)
    next_id = allocator.next()  # Returns 1000, then 1001, etc.
    peek_id = allocator.peek()  # Returns the next ID without incrementing it.
    allocator.reset(2000)  # Resets the next ID to 2000.
    ```
    """
    def __init__(self, start: int = 1000) -> None:
        self._next_id = start

    def next(self) -> int:
        current = self._next_id
        self._next_id += 1
        return current

    def peek(self) -> int:
        return self._next_id

    def reset(self, to: int) -> None:
        self._next_id = to


def infer_dtype(value: Any) -> str:
    """
    Infers the JSON schema type from a Python value.

    Args:
        value (Any): The value to infer from.

    Raises:
        TypeError: If the value is not supported by JSON schema.

    Returns:
        str: The JSON schema conform value.
    """
    if isinstance(value, dict):
        return "object"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "number"
    elif value is None:
        return "null"
    else:
        LOGGER.error(f"Unsupported data type: {type(value)}")
        raise TypeError(f"Unsupported data type: {type(value)}")


def select_general_dtype(current: str, previous: str) -> str:
    """
    Selects the data type based on the more general one.
    - `null` is always overwritten (possibly with `null`).
    - `number` is more general than `integer`.

    Args:
        current (str): The current data type.
        previous (str): The data type to compare the current with.

    Returns:
        str: The more general data type from the two provided.
    """
    if current == "null" or current == previous:
        return previous
    elif current == "number" and previous == "integer":
        return current
    elif previous == "number" and current == "integer":
        return previous
    else:
        return current


def infer_json_schema(data: Any, previous_schema: dict[str, Any] = {}, add_required: bool = False) -> Any:
    """
    Recursively generates a JSON schema for the given data structure.

    If the schema is unknown, is created from the first iteration by
    setting `previous_schema = {}`. The function can refine the schema
    using another dataset.

    Example:
    ```python
    schema = {}
    for ind in range(len(data_list)):
        schema = infer_json_schema(data_list[ind], schema, False)
    ```

    Args:
        data (Any): The data to generate a schema for.
        previous_schema (dict[str, Any]): The existing schema for the data.
        add_required (bool, optional): If set to true a `required` array is included
            in the schema for each `object` field. Defaults to False.

    Returns:
        Any: A JSON schema for the data.
    """
    if isinstance(data, dict):
        properties = {}
        required = []
        for key, value in data.items():
            if previous_schema \
                and "properties" in previous_schema \
                    and key in previous_schema["properties"]:
                sub_schema = previous_schema["properties"][key]
            else:
                sub_schema = {}
            properties[key] = infer_json_schema(value, sub_schema, add_required)
            required.append(key)
        result = {
            "type": "object",
            "properties": properties
        }
        if add_required:
            result["required"] = required
        return result
    elif isinstance(data, list):
        # Assuming all elements of the list have the same structure
        if len(data) > 0:
            dtypes = []
            for item in data:
                dtype = infer_dtype(item)
                if dtype == "object":
                    dtypes.append(infer_json_schema(item, previous_schema.get("items", {}), add_required))
                else:
                    dtypes.append(dtype)
            result_set: set[Any] = set([json.dumps(t) for t in dtypes])
            result_list: list[dict[str, Any]] = [json.loads(i) for i in result_set]
            return {
                "type": "array",
                "items": {} if len(result_list) == 0 else (result_list[0] if len(result_list) == 1 else result_list)
            }
        else:
            return {
                "type": "array",
                "items": {}
            }
    else:
        if previous_schema and "type" in previous_schema:
            previous_type = previous_schema["type"]
            data_type = select_general_dtype(infer_dtype(data), previous_type)
        else:
            data_type = infer_dtype(data)

        return {"type": data_type}


def infer_json_schema_from_dataset(data: list[dict[str, str]], schema: dict[str, str] = {}, add_required: bool = False) -> dict[str, Any]:
    """
    Generates a JSON schema from an array of dictionaries.

    Args:
        data_array (list): A list of dictionaries containing nested data structures.
        schema (dict[str, str]): A schema dictaionary to use, defaults to an empty dictionary.
        add_required (bool, optional): If set to true a `required` array is included
            in the schema for each `object` field. Defaults to False.
    Returns:
        dict: The generated and refined JSON schema.
    """
    if not data or not isinstance(data, list):
        LOGGER.error("Input must be a non-empty list of dictionaries.")
        raise ValueError("Input must be a non-empty list of dictionaries.")

    for ind in range(len(data)):
        schema = infer_json_schema(data[ind], schema, add_required)

    return schema


def flatten_schema(schema: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Flattens the provided JSON schema.
    - Removes keywords like `type` and `properties`.
    - Handles nested fields by separating them by `.`.

    Args:
        schema (dict[str, Any]): A dictionary in JSON schema format.
        parent_key (str, optional): Used in recursion. Defaults to "".
        sep (str, optional): Separator to use when unnesting fields. Defaults to ".".

    Returns:
        dict[str, Any]: A flattened dictionary.

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
    items = {}
    for k, v in schema.get("properties", {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if v.get("type") == "object":
            items.update(flatten_schema(v, new_key, sep=sep))
        else:
            items[new_key] = v["type"]
    return items


def build_iceberg_struct_type(properties: dict[str, Any], allocator: "IcebergIDAllocator", required_fields: list[str] = []):
    """
    Helper to convert a JSON schema `properties` dict into an Iceberg StructType.
    Uses the converter instance to ensure correct struct field parsing.
    """
    fields = [
        parse_iceberg_field(field_name, field_def, allocator, required_fields)
        for field_name, field_def in properties.items()
    ]
    return IcebergStructType(*fields)


def parse_iceberg_field(
    name: str,
    field_definition: dict[str, Any],
    allocator: "IcebergIDAllocator",
    required_fields: list[str] = []
) -> IcebergNestedField:
    """
    Parses a single JSON schema field into an Iceberg NestedField.
    Args:
        name (str): Field name.
        field_definition (dict[str, Any]): Field definition from schema.
        allocator (IcebergIDAllocator): ID allocator.
        required_fields (list[str]): List of required field names.
    Returns:
        IcebergNestedField: The parsed Iceberg NestedField.
    """
    field_type = field_definition.get("type")
    is_required = name in required_fields

    # We assign a synthetic name to the list/map element solely to pass through the parse function,
    # which expects a field name for ID generation. This name is not persisted in the schema,
    # as Iceberg list/map elements are anonymous in the final schema definition.
    if field_type == "object":
        result_type = build_iceberg_struct_type(
            field_definition.get("properties", {}),
            allocator,
            field_definition.get("required", required_fields)
        )
    elif field_type == "array":
        result_type = IcebergListType(
            element_id=allocator.next(),
            element_type=parse_iceberg_field(f"{name}_element", field_definition["items"], allocator, []).field_type,
            element_required=False
        )
    elif field_type == "map":
        result_type = IcebergMapType(
            key_id=allocator.next(),
            key_type=parse_iceberg_field(f"{name}_key", field_definition["properties"]["key"], allocator, []).field_type,
            value_id=allocator.next(),
            value_type=parse_iceberg_field(f"{name}_value", field_definition["properties"]["value"], allocator, []).field_type,
            value_required=False
        )
    elif field_type == "decimal":
        precision = int(field_definition["properties"]["precision"])
        scale = int(field_definition["properties"]["scale"])
        result_type = IcebergDecimalType(precision, scale)
    elif field_type in ICEBERG_TYPE_MAP:
        result_type = ICEBERG_TYPE_MAP[field_type]()
    else:
        raise ValueError(f"Unsupported type '{field_type}' in field '{name}'")

    return IcebergNestedField(field_id=allocator.next(), name=name, field_type=result_type, required=is_required)
