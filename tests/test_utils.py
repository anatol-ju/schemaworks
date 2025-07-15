import json
import pytest
from decimal import Decimal

from schemaworks.utils import (
    DecimalEncoder,
    IcebergIDAllocator,
    infer_dtype,
    parse_iceberg_field,
    select_general_dtype,
    infer_json_schema,
    infer_json_schema_from_dataset,
    build_iceberg_struct_type,
    flatten_schema
)
from pyiceberg.types import (
    IntegerType as IcebergIntegerType,
    StringType as IcebergStringType,
    DecimalType as IcebergDecimalType,
    MapType as IcebergMapType,
    ListType as IcebergListType,
    StructType as IcebergStructType,
    NestedField as IcebergNestedField
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


def test_infer_json_schema_simple_dict():
    """Generate JSON schema for a simple dictionary without required flag."""
    data = {"x": 1, "y": "foo"}
    schema = infer_json_schema(data, {}, add_required=False)
    assert schema == {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "string"}
        }
    }


def test_infer_json_schema_with_required_flag():
    """Generate JSON schema for a dictionary with required flag."""
    data = {"x": 1}
    schema = infer_json_schema(data, {}, add_required=True)
    assert schema["type"] == "object"
    assert schema["properties"]["x"] == {"type": "integer"}
    assert schema["required"] == ["x"]


def test_infer_json_schema_list_of_primitives():
    """Generate JSON schema for a list of primitives."""
    data = [1, 2, 3]
    schema = infer_json_schema(data, {}, add_required=False)
    assert schema == {"type": "array", "items": "integer"}


def test_infer_json_schema_list_of_objects():
    """Generate JSON schema for a list of objects."""
    data = [{"a": 1}, {"a": 2}]
    schema = infer_json_schema(data, {}, add_required=False)
    assert schema["type"] == "array"
    assert isinstance(schema["items"], dict)
    assert schema["items"]["type"] == "object"
    assert "properties" in schema["items"]


def test_infer_json_schema_empty_list():
    """Generate JSON schema for an empty list."""
    data = []
    schema = infer_json_schema(data, {}, add_required=False)
    assert schema == {"type": "array", "items": {}}


def test_infer_json_schema_respects_previous_schema():
    """Respect previous schema types during schema generation."""
    data = {"a": 5}
    prev = {"properties": {"a": {"type": "number"}}}
    schema = infer_json_schema(data, prev, add_required=False)
    assert schema["properties"]["a"]["type"] == "number"


def test_infer_json_schema_from_dataset_invalid_inputs():
    """Raise ValueError for invalid inputs in infer_json_schema_from_dataset."""
    with pytest.raises(ValueError):
        infer_json_schema_from_dataset([], {})
    with pytest.raises(ValueError):
        infer_json_schema_from_dataset("not a list", {})


def test_infer_json_schema_from_dataset_basic():
    """Generate JSON schema from a list of dictionaries correctly."""
    data = [{"k": "v"}, {"k": "w"}]
    schema = infer_json_schema_from_dataset(data, {}, add_required=False)
    assert schema["type"] == "object"
    assert "properties" in schema and "k" in schema["properties"]


def test_infer_json_schema_from_dataset_with_required():
    """Include required fields when generating JSON schema."""
    data = [{"k": "v"}]
    schema = infer_json_schema_from_dataset(data, {}, add_required=True)
    assert schema.get("required") == ["k"]


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
    assert flatten_schema(schema) == expected_output


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
    assert flatten_schema(schema, sep="/") == expected_output


def test_flatten_schema_with_no_properties() -> None:
    """Return empty dict when flattening schema with no properties."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "example-schema"
    }
    expected_output = {}
    assert flatten_schema(schema) == expected_output


def test_parse_primitive_field():
    """Parse a primitive field type correctly."""
    allocator = IcebergIDAllocator(start=1)
    field = parse_iceberg_field("age", {"type": "integer"}, allocator)
    assert isinstance(field, IcebergNestedField)
    assert field.name == "age"
    assert isinstance(field.field_type, IcebergIntegerType)
    assert field.field_id == 1  # ID is assigned during allocation


def test_iceberg_id_allocator_peek():
    """Test the peek functionality of IcebergIDAllocator."""
    allocator = IcebergIDAllocator(start=42)
    assert allocator.peek() == 42
    _ = allocator.next()
    assert allocator.peek() == 43
    allocator.reset(100)
    assert allocator.peek() == 100


def test_parse_required_field():
    """Parse a required field type correctly."""
    allocator = IcebergIDAllocator(start=10)
    field = parse_iceberg_field("name", {"type": "string"}, allocator, required_fields=["name"])
    assert field.required is True


def test_parse_array_field():
    """Parse an array field type correctly."""
    allocator = IcebergIDAllocator(start=100)
    schema = {"type": "array", "items": {"type": "string"}}
    field = parse_iceberg_field("tags", schema, allocator)
    assert isinstance(field.field_type, IcebergListType)
    assert isinstance(field.field_type.element_type, IcebergStringType)


def test_parse_map_field():
    """Parse a map field type correctly."""
    allocator = IcebergIDAllocator(start=200)
    schema = {
        "type": "map",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "integer"}
        }
    }
    field = parse_iceberg_field("attributes", schema, allocator)
    assert isinstance(field.field_type, IcebergMapType)
    assert isinstance(field.field_type.key_type, IcebergStringType)
    assert isinstance(field.field_type.value_type, IcebergIntegerType)


def test_parse_decimal_field():
    """Parse a decimal field type correctly."""
    allocator = IcebergIDAllocator(start=300)
    schema = {
        "type": "decimal",
        "properties": {
            "precision": 10,
            "scale": 2
        }
    }
    field = parse_iceberg_field("price", schema, allocator)
    assert isinstance(field.field_type, IcebergDecimalType)
    assert field.field_type.precision == 10
    assert field.field_type.scale == 2


def test_parse_unsupported_field_type():
    """Raise ValueError for unsupported field types."""
    allocator = IcebergIDAllocator(start=500)
    schema = {"type": "unsupported_type"}
    with pytest.raises(ValueError) as excinfo:
        parse_iceberg_field("foo", schema, allocator)
    assert "Unsupported type 'unsupported_type' in field 'foo'" in str(excinfo.value)


def test_to_iceberg_struct_nested_object():
    """Convert a nested object schema to IcebergStructType."""
    allocator = IcebergIDAllocator(start=1000)
    properties = {
        "user": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"}
            },
            "required": ["id"]
        }
    }
    struct = build_iceberg_struct_type(properties, allocator)
    assert isinstance(struct, IcebergStructType)
    user_field = struct.fields[0]
    assert user_field.name == "user"
    assert isinstance(user_field.field_type, IcebergStructType)
    subfields = user_field.field_type.fields
    assert len(subfields) == 2
    assert any(f.name == "id" and f.required for f in subfields)
