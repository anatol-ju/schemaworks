import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

import jsonschema.validators
import numpy as np
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from jsonschema.protocols import Validator


# Custom type check functions
def _is_float(value: Any) -> bool:
    return isinstance(value, (float, np.floating))

def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))

def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer))

def _is_long(value: Any) -> bool:
    return isinstance(value, (int, np.int64))

def _is_date(value: Any) -> bool:
    # Value can be a date string, datetime object, or numpy datetime64
    try:
        dt.date.fromisoformat(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return isinstance(value, (dt.date, np.datetime64))

def _is_datetime(value: Any) -> bool:
    try:
        dt.datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return isinstance(value, (dt.datetime, np.datetime64))

def _is_time(value: Any) -> bool:
    try:
        dt.time.fromisoformat(value)
        return True
    except ValueError:
        return False
    except TypeError:
        return isinstance(value, (dt.time, np.datetime64))

def _is_array(value: Any) -> bool:
    return isinstance(value, (list, np.ndarray))

def _is_object(value: Any) -> bool:
    return isinstance(value, dict)

def _is_map(value: Any) -> bool:
    return isinstance(value, dict)

def _extended_type_checker(
    validator: Validator, custom_type: str, instance: Any, schema: Any
) -> Any:
    checkers = {
        "number": _is_float,
        "float": _is_float,
        "boolean": _is_bool,
        "bool": _is_bool,
        "integer": _is_int,
        "int": _is_int,
        "long": _is_long,
        "date": _is_date,
        "datetime": _is_datetime,
        "time": _is_time,
        "array": _is_array,
        "object": _is_object,
        "map": _is_map
    }
    if custom_type in checkers:
        if not checkers[custom_type](instance):
            yield ValidationError(f"'{instance}' is of type '{type(instance)}' but should be '{custom_type}'.")
    else:
        for error in Draft202012Validator.VALIDATORS["type"](validator, custom_type, instance, schema):
            yield error


# Extend the existing validator
JsonSchemaValidator = jsonschema.validators.extend(
    Draft202012Validator,
    validators={"type": _extended_type_checker}
)
"""
The protocol to which all validator classes adhere.

Args:
    schema (Mapping[str, Any] | bool):
        The schema that the validator object will validate with. It is assumed to be valid,
        and providing an invalid schema can lead to undefined behavior.
        See `Validator.check_schema` to validate a schema first.
    registry (referencing.jsonschema.SchemaRegistry):
        A schema registry that will be used for looking up JSON references.
    format_checker (jsonschema.FormatChecker | None):
        If provided, a checker which will be used to assert about
        format properties present in the schema. If unprovided, *no* format validation is done,
        and the presence of format within schemas is strictly informational.
        Certain formats require additional packages to be installed in order to assert against
        instances. Ensure you've installed jsonschema with its extra (optional) dependencies
        when invoking pip.
"""


class BaseTypeValidator(ABC):
    """
    Abstract base class to create validators for different data types.

    Properties to implement:
        - default_values (dict[str, Any]): Map between types defined in schema and values
            to be used when the data value is `None`.
        - dtype_map (dict[str, Any]): A map to convert types from schema to the target data type.
    """
    @property
    @abstractmethod
    def default_values(self) -> dict[str, Any]:  # pragma: no cover
        pass

    @default_values.setter
    @abstractmethod
    def default_values(self, value: dict[str, Any]) -> None:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def dtype_map(self) -> dict[str, Any]:  # pragma: no cover
        pass

    @dtype_map.setter
    @abstractmethod
    def dtype_map(self, value: dict[str, Any]) -> None:  # pragma: no cover
        pass

    def conform(self, data: Any, schema: dict[str, Any], fill_missing: bool = False, fill_nested: bool = False) -> Any:
        """
        Conforms data to a specified schema.
        For this, the function recursively traverses through the schema and casts the values
        of the data to the ones specified in the schema.

        If a `None` type is found, it's replaced by a default value of the type specified
        in the schema:
        - any `0` values are converted into floats if the expected type is `float`
        - any `0.0` values are converted to integers if the expected type is `integer`.
        - `boolean` type default is False
        - `array` type default is an empty array
        - `object` or `map` defaults are empty dictionaries
        - `date` and `datetime` has `00:00:00` as default value

        If fields are missing in the data that are defined in the schema, they can be added with
        a default value based on the data type. Set `fill_missing` to `True` to enable this.

        Use the `fill_nested` option to control the validation depth. Use this, if it is sufficient
        to know the top-level structure (like dict or list) without validating the nested fields.
        Setting `fill_nested = False` means, whenever the data is empty (like `[]` or `{}`), the
        validator doesn't check for nested fields and therefore doesn't fill the missing ones.

        Args:
            data (Any): The data to be validated, usually a dictionary.
            schema (dict[str, Any]): The schema to validate the data against, as a dictionary.
            fill_missing (bool, optional): If `True`, any fields that are defined in the schema but are
                missing in the data are included to the result with their default values from
                the schema. Defaults to `False`.
            fill_nested (bool, optional): If `True`, nested structures are also filled if they are empty.
                For example: `data = None` but a nested field `key` is required by the schema,
                then `key` is added like `data = {"key": 0}` (with the appropriate default value).
                This parameter is ignored if `fill_missing` is set to `False`.
                Defaults to `False`.

        Returns:
            Any: The result is the validated dataset.
        """

        # 'type' is not a required keyword at top level
        if "type" not in schema and "properties" in schema:
            schema["type"] = "object"
            return self.conform(data, schema, fill_missing=fill_missing, fill_nested=fill_nested)

        if schema["type"] == "map" and data:
            result_dict = {}
            for key, value in schema["properties"].items():
                k = self.conform(list(data.keys())[0], value, fill_missing=fill_missing, fill_nested=fill_nested)
                v = self.conform(list(data.values())[0], value, fill_missing=fill_missing, fill_nested=fill_nested)
                result_dict[k] = v
            return result_dict

        elif schema["type"] == "object" and data:
            result_dict = {}
            for key, value in schema["properties"].items():
                # Expect missing keys in data
                if key in data:
                    result_dict[key] = self.conform(data[key], value, fill_missing=fill_missing, fill_nested=fill_nested)
                else:
                    if fill_missing:
                        result_dict[key] = self.conform(None, value, fill_missing=fill_missing, fill_nested=fill_nested)

            extra_keys = [key for key in data if key not in schema["properties"].keys()]
            for extra_key in extra_keys:
                result_dict[extra_key] = data[extra_key]
            return result_dict

        elif schema["type"] == "array" and data:
            result_list = []
            for item in data:
                result_list.append(self.conform(item, schema["items"], fill_missing=fill_missing, fill_nested=fill_nested))
            return result_list

        else:
            # handle non-nested values
            if not data or data == 0:
                if fill_nested and "properties" in schema and len(schema["properties"]) > 0:
                    result = {}
                    for key, value in schema["properties"].items():
                        result[key] = self.conform(None, value, fill_missing=fill_missing, fill_nested=fill_nested)
                    return result
                return self.default_values.get(schema["type"], None)
            else:
                dtype = self.dtype_map[schema["type"]]
                return dtype(data)

    def validate(self, data: Any, schema: dict[str, Any]) -> None:
        """
        Validate a dataset against a schema.
        A custom validator based on the jsonschema.Draft202012Validator is used
        and extended to validate the following types:
        - number
        - float
        - boolean
        - bool
        - integer
        - int
        - long
        - date
        - datetime
        - time
        - array
        - object
        - map

        Args:
            data (Any): The data to validate.
            schema (dict[str, Any]): The JSON schema to use for validation.
        """
        validator = JsonSchemaValidator(schema=schema)
        validator.validate(data)


# Concrete class implementing the abstract class
class PythonTypeValidator(BaseTypeValidator):
    """
    A validator for Python data types.
    """
    def __init__(self) -> None:
        self._default_values = {
            "string": "",
            "number": 0.0,
            "float": 0.0,
            "integer": 0,
            "int": 0,
            "long": 0,
            "boolean": False,
            "bool": False,
            "array": [],
            "object": {},
            "map": {},
            "date": dt.date.min,
            "datetime": dt.datetime.min,
            "time": dt.time.min
        }
        self._dtype_map = {
            "string": str,
            "number": float,
            "float": float,
            "integer": int,
            "int": int,
            "long": int,
            "boolean": bool,
            "bool": bool,
            "array": list
        }

    @property
    def default_values(self) -> dict[str, Any]:  # pragma: no cover
        return self._default_values

    @default_values.setter
    def default_values(self, value: dict[str, Any]) -> None:  # pragma: no cover
        self._default_values = value

    @property
    def dtype_map(self) -> dict[str, Any]:  # pragma: no cover
        return self._dtype_map

    @dtype_map.setter
    def dtype_map(self, value: dict[str, Any]) -> None:  # pragma: no cover
        self._dtype_map = value


class NumpyTypeValidator(BaseTypeValidator):
    """
    A validator for NumPy data types.
    """
    def __init__(self) -> None:
        self._default_values = {
            "string": np.str_(""),
            "number": np.float32(0.0),
            "float": np.float32(0.0),
            "integer": np.int32(0),
            "int": np.int32(0),
            "long": np.int64(0),
            "boolean": np.bool_(False),
            "bool": np.bool_(False),
            "array": [],
            "object": {},
            "map": {},
            "date": np.datetime64("1678-01-01", "D"),
            "datetime": np.datetime64("1678-01-01T00:00:00", "s"),
            "time": np.datetime64("1678-01-01T00:00:00", "s")
        }
        self._dtype_map = {
            "string": np.str_,
            "number": np.float32,
            "float": np.float32,
            "integer": np.int32,
            "int": np.int32,
            "long": np.int64,
            "boolean": np.bool_,
            "bool": np.bool_,
            "array": np.ndarray
        }

    @property
    def default_values(self) -> dict[str, Any]:  # pragma: no cover
        return self._default_values

    @default_values.setter
    def default_values(self, value: dict[str, Any]) -> None:  # pragma: no cover
        self._default_values = value

    @property
    def dtype_map(self) -> dict[str, Any]:  # pragma: no cover
        return self._dtype_map

    @dtype_map.setter
    def dtype_map(self, value: dict[str, Any]) -> None:  # pragma: no cover
        self._dtype_map = value
