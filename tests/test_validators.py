import datetime as dt
from typing import Any, Optional, Union

import numpy as np
import pytest
from jsonschema import ValidationError

from schemaworks.validators import (
    JsonSchemaValidator,
    NumpyTypeValidator,
    PythonTypeValidator,
    _is_array,
    _is_bool,
    _is_date,
    _is_datetime,
    _is_float,
    _is_int,
    _is_long,
    _is_map,
    _is_object,
    _is_time
)

# Define a schema containing all possible data types
schema: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "mysense-test-schema",
    "type": "object",
    "properties": {
        "string_prop": {"type": "string"},
        "number_prop": {"type": "number"},
        "integer_prop": {"type": "integer"},
        "float_prop": {"type": "float"},
        "boolean_prop": {"type": "boolean"},
        "long_prop": {"type": "long"},
        "date_prop": {"type": "date"},
        "datetime_prop": {"type": "datetime"},
        "time_prop": {"type": "time"},
        "array_prop": {"type": "array", "items": {"type": "string"}},
        "object_prop": {
            "type": "object",
            "properties": {"bool_prop": {"type": "bool"}},
        },
        "map_prop": {
            "type": "map",
            "properties": {"key": {"type": "string"}, "value": {"type": "bool"}},
        },
    },
}

# Test cases for valid inputs
valid_instances: list[dict[str, Any]] = [
    {"string_prop": "test"},
    {"number_prop": 123.456},
    {"integer_prop": 123},
    {"float_prop": 1.0},
    {"boolean_prop": True},
    {"long_prop": 1000000},
    {"date_prop": dt.date(2020, 1, 1)},
    {"datetime_prop": dt.datetime(2020, 1, 1, 0, 0, 0)},
    {"time_prop": dt.time(0, 0, 0)},
    {"array_prop": ["item1", "item2"]},
    {"object_prop": {"bool_prop": False}},
    {"map_prop": {"key_string": False, "other_string": True}},
]

# Test cases for invalid inputs
invalid_instances: list[dict[str, Any]] = [
    {"string_prop": 123},
    {"number_prop": "not_a_number"},
    {"integer_prop": "not_an_integer"},
    {"float_prop": "not_a_float"},
    {"boolean_prop": "not_a_bool"},
    {"long_prop": "not_a_long"},
    {"date_prop": "not_a_date"},
    {"datetime_prop": "not_a_datetime"},
    {"time_prop": "not_a_time"},
    {"array_prop": ["item1", 2]},
    {"object_prop": {"bool_prop": "not_a_bool"}},
    {"map_prop": {"key": "some_key", "value": "not_a_bool"}},
]


@pytest.fixture
def numpy_validator() -> NumpyTypeValidator:
    return NumpyTypeValidator()


@pytest.fixture
def python_validator() -> PythonTypeValidator:
    return PythonTypeValidator()


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.23, True),
        (np.float32(1.23), True),
        ("1.23", False),
        (123, False),
    ],
)
def test_is_float(value: Any, expected: bool) -> None:
    """
    Test that _is_float returns True for float types and False for non-floats.
    """
    assert _is_float(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, True),
        (np.bool_(True), True),
        (1, False),
        ("True", False),
    ],
)
def test_is_bool(value: Any, expected: bool) -> None:
    """
    Test that _is_bool returns True for boolean types and False otherwise.
    """
    assert _is_bool(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (123, True),
        (np.int32(123), True),
        (1.23, False),
        ("123", False),
    ],
)
def test_is_int(value: Any, expected: bool) -> None:
    """
    Test that _is_int returns True for integer types and False for non-integers.
    """
    assert _is_int(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.int64(123), True),
        (123, True),
        (1.23, False),
        ("123", False),
    ],
)
def test_is_long(value: Any, expected: bool) -> None:
    """
    Test that _is_long returns True for long integer types and False otherwise.
    """
    assert _is_long(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2023-01-01", True),
        (dt.date(2023, 1, 1), True),
        (np.datetime64("2023-01-01"), True),
        ("not_a_date", False),
        (123, False),
    ],
)
def test_is_date(value: Any, expected: bool) -> None:
    """
    Test that _is_date recognizes date inputs and rejects invalid values.
    """
    assert _is_date(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2023-01-01T12:00:00", True),
        (dt.datetime(2023, 1, 1, 12, 0, 0), True),
        (np.datetime64("2023-01-01T12:00:00"), True),
        ("not_a_datetime", False),
        (123, False),
    ],
)
def test_is_datetime(value: Any, expected: bool) -> None:
    """
    Test that _is_datetime recognizes datetime inputs and rejects invalid values.
    """
    assert _is_datetime(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12:00:00", True),
        (dt.time(12, 0, 0), True),
        (np.datetime64("2023-01-01T12:00:00"), True),
        ("not_a_time", False),
        (123, False),
    ],
)
def test_is_time(value: Any, expected: bool) -> None:
    """
    Test that _is_time recognizes time inputs and rejects invalid values.
    """
    assert _is_time(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ([1, 2, 3], True),
        (np.array([1, 2, 3]), True),
        ("not_an_array", False),
        (123, False),
    ],
)
def test_is_array(value: Any, expected: bool) -> None:
    """
    Test that _is_array returns True for array-like types and False otherwise.
    """
    assert _is_array(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"key": "value"}, True),
        ([], False),
        ("not_an_object", False),
        (123, False),
    ],
)
def test_is_object(value: Any, expected: bool) -> None:
    """
    Test that _is_object returns True for dict objects and False for other types.
    """
    assert _is_object(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ({"key": "value"}, True),
        ([], False),
        ("not_a_map", False),
        (123, False),
    ],
)
def test_is_map(value: Any, expected: bool) -> None:
    """
    Test that _is_map returns True for map-like dicts and False otherwise.
    """
    assert _is_map(value) == expected


@pytest.mark.parametrize("instance", valid_instances)
def test_valid_instances(instance: dict[str, Any]) -> None:
    """
    Test that JsonSchemaValidator.validate accepts all valid instance data.
    """
    validator = JsonSchemaValidator(schema)
    validator.validate(instance)


@pytest.mark.parametrize("instance", invalid_instances)
def test_invalid_instances(instance: dict[str, Any]) -> None:
    """
    Test that JsonSchemaValidator.validate raises ValidationError for invalid instances.
    """
    validator = JsonSchemaValidator(schema)
    with pytest.raises(ValidationError):
        validator.validate(instance)


@pytest.mark.parametrize("validator", [PythonTypeValidator(), NumpyTypeValidator()])
def test_validate_scalar_types(validator):
    """
    Test that validators accept valid scalar types and reject incorrect types.
    """
    # should accept valid scalars
    validator.validate(123, {"type": "integer"})
    validator.validate("hello", {"type": "string"})
    validator.validate(True, {"type": "boolean"})

    # should reject invalid scalars
    with pytest.raises(ValidationError):
        validator.validate("not an int", {"type": "integer"})
    with pytest.raises(ValidationError):
        validator.validate(5, {"type": "boolean"})


def test_validate_complex_types(python_validator):
    """
    Test array, object and map validation in the type validators.
    """
    # arrays
    python_validator.validate(
        [1, 2, 3], {"type": "array", "items": {"type": "integer"}}
    )
    with pytest.raises(ValidationError):
        python_validator.validate(
            [1, "a"], {"type": "array", "items": {"type": "integer"}}
        )

    # objects
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
    }
    python_validator.validate({"a": 1, "b": "x"}, schema)
    with pytest.raises(ValidationError):
        python_validator.validate({"a": 1, "b": 2}, schema)

    # maps (same as object but keys/values validated)
    schema = {
        "type": "map",
        "properties": {"key": {"type": "string"}, "value": {"type": "integer"}},
    }
    python_validator.validate({"foo": 42}, schema)  # key "foo" ok, value 42 ok
    # it's a map, all key/value pairs are accepted
    python_validator.validate({1: "bad key"}, schema)


def test_conform_with_python_type_validator(
    python_validator: PythonTypeValidator,
) -> None:
    """
    Test PythonTypeValidator.conform fills missing object fields with defaults.
    """
    data: dict[str, Optional[int]] = {"name": None, "age": 0, "is_active": None}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
    }
    expected_output: dict[str, Union[str, int, bool]] = {
        "name": "",
        "age": 0,
        "is_active": False,
    }
    assert python_validator.conform(data, schema, fill_missing=True) == expected_output


def test_conform_with_numpy_type_validator(numpy_validator: NumpyTypeValidator) -> None:
    """
    Test NumpyTypeValidator.conform fills missing object fields with numpy defaults.
    """
    data: dict[str, Optional[int]] = {"name": None, "age": 0, "is_active": None}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        },
    }
    expected_output: dict[str, Union[str, int, bool]] = {
        "name": "",
        "age": 0,
        "is_active": False,
    }
    assert numpy_validator.conform(data, schema, fill_missing=True) == expected_output


def test_conform_without_type_field(python_validator: PythonTypeValidator) -> None:
    """
    Test conform behavior when schema properties lack a type field.
    """
    data: dict[str, Optional[int]] = {"name": None, "age": 0, "is_active": None}
    schema: dict[str, Any] = {
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "is_active": {"type": "boolean"},
        }
    }
    expected_output: dict[str, Union[str, int, bool]] = {
        "name": "",
        "age": 0,
        "is_active": False,
    }
    assert python_validator.conform(data, schema, fill_missing=True) == expected_output


def test_conform_with_missing_fields(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that missing object properties are added with default values when fill_missing is True.
    """
    data: dict[str, Union[str, None]] = {"name": "John"}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    expected_output: dict[str, Union[str, int]] = {"name": "John", "age": 0}
    assert numpy_validator.conform(data, schema, fill_missing=True) == expected_output
    assert python_validator.conform(data, schema, fill_missing=True) == expected_output


def test_conform_with_missing_nested_fields(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test deep nested property filling with fill_missing and deep flags.
    """
    data: dict[str, Optional[int]] = {"name": None}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "object",
                "properties": {
                    "first": {"type": "string"},
                    "last": {"type": "string"},
                    "middle": {"type": "boolean"},
                },
            },
            "age": {"type": "integer"},
        },
    }
    expected_output: dict[str, Union[dict[str, Union[str, bool]], int]] = {
        "name": {"first": "", "last": "", "middle": False},
        "age": 0,
    }
    assert (
        numpy_validator.conform(data, schema, fill_missing=True, fill_nested=True)
        == expected_output
    )
    assert (
        python_validator.conform(data, schema, fill_missing=True, fill_nested=True)
        == expected_output
    )


def test_conform_array_type(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that array data is returned unchanged when fill_missing is False.
    """
    data: list[int] = [1, 2, 3]
    schema: dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
    expected_output: list[int] = [1, 2, 3]
    assert numpy_validator.conform(data, schema, fill_missing=False) == expected_output
    assert python_validator.conform(data, schema, fill_missing=False) == expected_output


def test_conform_map_type(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that map data is returned unchanged when fill_missing is False.
    """
    data: dict[str, str] = {"key": "value"}
    schema: dict[str, Any] = {
        "type": "map",
        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
    }
    expected_output: dict[str, str] = {"key": "value"}
    assert numpy_validator.conform(data, schema, fill_missing=False) == expected_output
    assert python_validator.conform(data, schema, fill_missing=False) == expected_output


def test_conform_string(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns empty string default for string type.
    """
    schema: dict[str, Any] = {"type": "string"}
    data: None = None
    assert numpy_validator.conform(data, schema) == ""
    assert isinstance(numpy_validator.conform(data, schema), np.str_)
    assert python_validator.conform(data, schema) == ""


def test_conform_number(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns zero float default for number type.
    """
    schema: dict[str, Any] = {"type": "number"}
    data: None = None
    assert numpy_validator.conform(data, schema) == 0.0
    assert isinstance(numpy_validator.conform(data, schema), np.float32)
    assert python_validator.conform(data, schema) == 0.0


def test_conform_integer(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns zero integer default for integer type.
    """
    schema: dict[str, Any] = {"type": "integer"}
    data: None = None
    assert numpy_validator.conform(data, schema) == 0
    assert isinstance(numpy_validator.conform(data, schema), np.int32)
    assert python_validator.conform(data, schema) == 0


def test_conform_boolean(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns False for boolean type default.
    """
    schema: dict[str, Any] = {"type": "boolean"}
    data: None = None
    assert numpy_validator.conform(data, schema) is np.bool_(False)
    assert isinstance(numpy_validator.conform(data, schema), np.bool_)
    assert python_validator.conform(data, schema) is False


def test_conform_date(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns the minimum date for date type default.
    """
    schema: dict[str, Any] = {"type": "date"}
    data: None = None
    assert numpy_validator.conform(data, schema) == np.datetime64("1678-01-01", "D")
    assert isinstance(numpy_validator.conform(data, schema), np.datetime64)
    assert python_validator.conform(data, schema) == dt.date.min


def test_conform_datetime(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns the minimum datetime for datetime type default.
    """
    schema: dict[str, Any] = {"type": "datetime"}
    data: None = None
    assert numpy_validator.conform(data, schema) == np.datetime64(
        "1678-01-01T00:00:00", "s"
    )
    assert isinstance(numpy_validator.conform(data, schema), np.datetime64)
    assert python_validator.conform(data, schema) == dt.datetime.min


def test_conform_time(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns the minimum time for time type default.
    """
    schema: dict[str, Any] = {"type": "time"}
    data: None = None
    assert numpy_validator.conform(data, schema) == np.datetime64(
        "1678-01-01T00:00:00", "s"
    )
    assert isinstance(numpy_validator.conform(data, schema), np.datetime64)
    assert python_validator.conform(data, schema) == dt.time.min


def test_conform_array(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns empty list default for array type.
    """
    schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    data: None = None
    assert numpy_validator.conform(data, schema) == []
    assert python_validator.conform(data, schema) == []


def test_conform_map(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that conform returns empty dict default for map type.
    """
    schema: dict[str, Any] = {
        "type": "map",
        "properties": {"key": {"type": "string"}, "value": {"type": "integer"}},
    }
    data: None = None
    assert numpy_validator.conform(data, schema, fill_missing=False) == {}
    assert python_validator.conform(data, schema, fill_missing=False) == {}


def test_conform_nested_dict(python_validator: PythonTypeValidator) -> None:
    """
    Test that nested objects are filled with default values correctly.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zipcode": {"type": "integer"},
                },
            },
        },
    }
    data: dict[str, Optional[dict[str, None]]] = {
        "name": None,
        "age": None,
        "address": {"city": None, "zipcode": None},
    }
    expected: dict[str, Union[str, int, dict[str, Union[str, int]]]] = {
        "name": "",
        "age": 0,
        "address": {"city": "", "zipcode": 0},
    }
    assert python_validator.conform(data, schema) == expected


def test_conform_nested_array(python_validator: PythonTypeValidator) -> None:
    """
    Test that nested arrays of objects are filled with default values correctly.
    """
    schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        },
    }
    data: list[dict[str, None]] = [
        {"name": None, "age": None},
        {"name": None, "age": None},
    ]
    expected: list[dict[str, Union[str, int]]] = [
        {"name": "", "age": 0},
        {"name": "", "age": 0},
    ]
    assert python_validator.conform(data, schema) == expected


def test_conform_integer_to_float_conversion(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that integer inputs are converted to floats for float type.
    """
    schema: dict[str, Any] = {"type": "float"}
    data: int = 0
    assert numpy_validator.conform(data, schema) == 0.0
    assert isinstance(numpy_validator.conform(data, schema), np.float32)
    assert python_validator.conform(data, schema) == 0.0


def test_conform_float_to_integer_conversion(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that float inputs are converted to integers for integer type.
    """
    schema: dict[str, Any] = {"type": "integer"}
    data: float = 0.0
    assert numpy_validator.conform(data, schema) == 0
    assert isinstance(numpy_validator.conform(data, schema), np.int32)
    assert python_validator.conform(data, schema) == 0


def test_conform_map_conversion(
    numpy_validator: NumpyTypeValidator, python_validator: PythonTypeValidator
) -> None:
    """
    Test that map keys and values are converted to correct types per schema.
    """
    schema: dict[str, Any] = {
        "type": "map",
        "properties": {"key": {"type": "integer"}, "value": {"type": "integer"}},
    }
    data: dict[float, float] = {1.0: 0.0}
    expected: dict[int, int] = {1: 0}
    assert numpy_validator.conform(data, schema) == expected
    assert python_validator.conform(data, schema) == expected


def test_conform_nested_dict_with_missing_fields(
    python_validator: PythonTypeValidator,
) -> None:
    """
    Test that nested missing fields are added when fill_missing is True.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zipcode": {"type": "integer"},
                },
            },
        },
    }
    data: dict[str, Optional[dict[str, None]]] = {
        "name": None,
        "address": {
            "city": None,
        },
    }
    expected: dict[str, Union[str, int, dict[str, Union[str, int]]]] = {
        "name": "",
        "age": 0,
        "address": {"city": "", "zipcode": 0},
    }
    assert python_validator.conform(data, schema, fill_missing=True) == expected


def test_conform_nested_dict_with_extra_fields(
    python_validator: PythonTypeValidator,
) -> None:
    """
    Test that extra fields in data are preserved when filling defaults.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zipcode": {"type": "integer"},
                },
            },
        },
    }
    data: dict[str, Optional[dict[str, Optional[int]]]] = {
        "name": None,
        "address": {
            "city": None,
        },
        "extra": None,
    }
    expected: dict[str, Union[str, int, dict[str, Union[str, int, None]], None]] = {
        "name": "",
        "age": 0,
        "address": {"city": "", "zipcode": 0},
        "extra": None,
    }
    assert python_validator.conform(data, schema, fill_missing=True) == expected
