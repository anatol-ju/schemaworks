import json
import pytest
from decimal import Decimal

from schemaworks.utils import (
    DecimalEncoder,
    infer_dtype,
    select_general_dtype,
    generate_schema,
    create_json_schema_from_data
)

def test_decimal_encoder_integer_decimal():
    """Convert Decimal with integer value to int correctly."""
    value = Decimal('5.0')
    encoded = DecimalEncoder().default(value)
    assert isinstance(encoded, int)
    assert encoded == 5

def test_decimal_encoder_non_integer_decimal():
    """Convert Decimal with fractional value to float correctly."""
    value = Decimal('3.14')
    encoded = DecimalEncoder().default(value)
    assert isinstance(encoded, float)
    assert encoded == pytest.approx(3.14)

def test_decimal_encoder_json_dumps():
    """Encode Decimals correctly using json.dumps."""
    obj = {'a': Decimal('2.0'), 'b': Decimal('4.5')}
    dumped = json.dumps(obj, cls=DecimalEncoder, sort_keys=True)
    assert dumped == '{"a": 2, "b": 4.5}'

def test_decimal_encoder_other_type_raises():
    """Raise TypeError for non-Decimal types in DecimalEncoder."""
    with pytest.raises(TypeError):
        DecimalEncoder().default(object())

@pytest.mark.parametrize("value, expected", [
    ({}, "object"),
    ([], "array"),
    ("hello", "string"),
    (42, "integer"),
    (3.0, "number"),
    (None, "null"),
    (False, "boolean"),
])
def test_infer_dtype_supported(value, expected):
    """Infer JSON schema type from supported Python values."""
    assert infer_dtype(value) == expected

def test_infer_dtype_unsupported():
    """Raise TypeError for unsupported types in infer_dtype."""
    class Foo:
        pass
    with pytest.raises(TypeError) as excinfo:
        infer_dtype(Foo())
    assert "Unsupported data type" in str(excinfo.value)

@pytest.mark.parametrize("current, previous, expected", [
    ("null", "string", "string"),
    ("string", "string", "string"),
    ("number", "integer", "number"),
    ("integer", "number", "number"),
    ("boolean", "integer", "boolean"),
])
def test_select_general_dtype(current, previous, expected):
    """Select the more general data type correctly."""
    assert select_general_dtype(current, previous) == expected

def test_generate_schema_simple_dict():
    """Generate JSON schema for a simple dictionary without required flag."""
    data = {"x": 1, "y": "foo"}
    schema = generate_schema(data, {}, add_required=False)
    assert schema == {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "string"}
        }
    }

def test_generate_schema_with_required_flag():
    """Generate JSON schema for a dictionary with required flag."""
    data = {"x": 1}
    schema = generate_schema(data, {}, add_required=True)
    assert schema["type"] == "object"
    assert schema["properties"]["x"] == {"type": "integer"}
    assert schema["required"] == ["x"]

def test_generate_schema_list_of_primitives():
    """Generate JSON schema for a list of primitives."""
    data = [1, 2, 3]
    schema = generate_schema(data, {}, add_required=False)
    assert schema == {"type": "array", "items": "integer"}

def test_generate_schema_list_of_objects():
    """Generate JSON schema for a list of objects."""
    data = [{"a": 1}, {"a": 2}]
    schema = generate_schema(data, {}, add_required=False)
    assert schema["type"] == "array"
    assert isinstance(schema["items"], dict)
    assert schema["items"]["type"] == "object"
    assert "properties" in schema["items"]

def test_generate_schema_empty_list():
    """Generate JSON schema for an empty list."""
    data = []
    schema = generate_schema(data, {}, add_required=False)
    assert schema == {"type": "array", "items": {}}

def test_generate_schema_respects_previous_schema():
    """Respect previous schema types during schema generation."""
    data = {"a": 5}
    prev = {"properties": {"a": {"type": "number"}}}
    schema = generate_schema(data, prev, add_required=False)
    assert schema["properties"]["a"]["type"] == "number"

def test_create_json_schema_from_data_invalid_inputs():
    """Raise ValueError for invalid inputs in create_json_schema_from_data."""
    with pytest.raises(ValueError):
        create_json_schema_from_data([], {})
    with pytest.raises(ValueError):
        create_json_schema_from_data("not a list", {})

def test_create_json_schema_from_data_basic():
    """Generate JSON schema from a list of dictionaries correctly."""
    data = [{"k": "v"}, {"k": "w"}]
    schema = create_json_schema_from_data(data, {}, add_required=False)
    assert schema["type"] == "object"
    assert "properties" in schema and "k" in schema["properties"]

def test_create_json_schema_from_data_with_required():
    """Include required fields when generating JSON schema."""
    data = [{"k": "v"}]
    schema = create_json_schema_from_data(data, {}, add_required=True)
    assert schema.get("required") == ["k"]
