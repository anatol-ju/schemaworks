import json
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


def generate_schema(data: Any, previous_schema: dict[str, Any], add_required: bool = False) -> Any:
    """
    Recursively generates a JSON schema for the given data structure.

    If the schema is unknown, is created from the first iteration by
    setting `previous_schema = {}`. The function can refine the schema
    using another dataset.

    Example:
    ```python
    schema = {}
    for ind in range(len(data_list)):
        schema = generate_schema(data_list[ind], schema, False)
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
            properties[key] = generate_schema(value, sub_schema, add_required)
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
                    dtypes.append(generate_schema(item, previous_schema.get("items", {}), add_required))
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


def create_json_schema_from_data(data: list[dict[str, str]], schema: dict[str, str], add_required: bool = False) -> dict[str, Any]:
    """
    Generates a JSON schema from an array of dictionaries.

    Args:
        data_array (list): A list of dictionaries containing nested data structures.
        schema (dict[str, str]): A schema dictaionary to use, can be empty if unknown.
        add_required (bool, optional): If set to true a `required` array is included
            in the schema for each `object` field. Defaults to False.
    Returns:
        dict: The generated and refined JSON schema.
    """
    if not data or not isinstance(data, list):
        raise ValueError("Input must be a non-empty list of dictionaries.")

    schema = {}
    for ind in range(len(data)):
        schema = generate_schema(data[ind], schema, add_required)

    return schema
