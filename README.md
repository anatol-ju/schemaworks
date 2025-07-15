# SchemaWorks

**SchemaWorks** is a Python library for converting between different schema definitions, such as JSON Schema, Spark DataTypes, SQL type strings, and more. It aims to simplify working with structured data across multiple data engineering and analytics platforms.

## 📣 New in 1.2.0
Added support to create Iceberg schemas to be used with PyIceberg.

## 🔧 Features

- Convert JSON Schema to:
  - Apache Spark StructType
  - SQL column type strings
  - Python dtypes dictionaries
  - Iceberg types (using PyIceberg)
- Convert Spark schemas and dtypes to JSON Schema
- Generate JSON Schemas from example data
- Flatten nested schemas for easier inspection or mapping
- Utilities for handling Decimal encoding and schema inference

## 🚀 Use Cases

- Building pipelines that consume or produce data in multiple formats
- Ensuring schema consistency across Spark, SQL, and data validation layers
- Automating schema generation from sample data for prototyping
- Simplifying developer tooling with schema introspection

## 🔍 Validation Support

SchemaWorks includes custom schema validation support through extended JSON Schema validators. It supports standard types like `string`, `integer`, `array`, and also recognises additional types common in data engineering workflows:

- Extended support for:
  - `float`, `bool`, `long`
  - `date`, `datetime`, `time`
  - `map`

Validation is performed using an enhanced version of `jsonschema.Draft202012Validator` that integrates these type checks.

## 🚚 Installation

You can install SchemaWorks using `pip` or `poetry`, depending on your preference.

### Using pip

Make sure you’re using Python 3.10 or later.

```bash
pip install schemaworks
```

This will install the package along with its core dependencies.

### Using Poetry

If you use [Poetry](https://python-poetry.org/) for dependency management:

```bash
poetry add schemaworks
```

To install development dependencies as well (for testing and linting):

```bash
poetry install --with dev
```

### Cloning the Repository (For Development)

If you want to clone and develop the package locally:

```bash
git clone https://github.com/anatol-ju/schemaworks.git
cd schemaworks
poetry install --with dev
pre-commit install  # optional: enable linting and formatting checks
```

To run the test suite:

```bash
poetry run pytest
```

## 🧱 Quick Example

```python
from schemaworks.converter import JsonSchemaConverter

# Load a JSON schema
schema = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "purchase": {
            "type": "object",
            "properties": {
                "item": {"type": "string"},
                "price": {"type": "number"}
            }
        }
    }
}

converter = JsonSchemaConverter(schema=schema)

# Convert to Spark schema
spark_schema = converter.to_spark_schema()
print(spark_schema)

# Convert to SQL string
sql_schema = converter.to_sql_string()
print(sql_schema)
```

## 📖 Documentation

- JSON ↔ Spark conversions
  Map JSON schema types to Spark StructTypes and back.
- Schema flattening
  Flatten nested schemas into dot notation for easier access and mapping.
- Data-driven schema inference
  Automatically generate JSON schemas from raw data samples.
- Decimal compatibility
  Custom JSON encoder to handle decimal.Decimal values safely.
- Schema validation
  Validate schemas and make data conform if needed.

## 🧪 Testing

Run unit tests using pytest:
```bash
poetry run pytest
```

## ⭐ Examples

### ✅ Convert JSON schema to Spark StructType

When working with data pipelines, it’s common to receive schemas in JSON format — whether from APIs, data contracts, or auto-generated metadata. But tools like Apache Spark and PySpark require their own schema definitions in the form of StructType. Manually translating between these formats is error-prone, time-consuming, and doesn’t scale. This function bridges that gap by automatically converting standard JSON Schemas into Spark-compatible schemas, saving hours of manual effort and reducing the risk of type mismatches in production pipelines.

```python
from schemaworks import JsonSchemaConverter

json_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string"},
        "price": {"type": "number"}
    }
}

converter = JsonSchemaConverter(schema=json_schema)
spark_schema = converter.to_spark_schema()
print(spark_schema)
```

### ✅ Infer schema from example JSON data

When working with dynamic or loosely structured data sources, manually writing a schema can be tedious and error-prone—especially when dealing with deeply nested or inconsistent inputs. This function allows you to infer a valid JSON Schema directly from real example data, making it much faster to prototype, validate, or document your datasets. It’s particularly useful when onboarding new datasets or integrating third-party APIs, where a formal schema may be missing or outdated.

```python
import json
from pprint import pprint
from schemaworks.utils import generate_schema

example_data = {}
with open("example_data.json", "r") as f:
    example_data = f.read()

example_data = json.loads(example_data)

schema = generate_schema(example_data, add_required=True)
pprint(schema)
```

### ✅ Flatten a nested schema

Flattening a nested JSON schema makes it easier to map fields to flat tabular structures, such as SQL tables or Spark DataFrames. It simplifies downstream processing, column selection, and validation—especially when working with deeply nested APIs or hierarchical datasets.

```python
converter.json_schema = {
    "type": "object",
    "properties": {
        "user_id": {"type": "integer"},
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "phone": {"type": "string"}
            },
        },
        "active": {"type": "boolean"},
    },
    "required": ["user_id", "email"],
}
flattened = converter.to_flat()
pprint(flattened)
```

### ✅ Convert inferred schema to SQL column types

After inferring or converting a schema, it's often necessary to express it in SQL-friendly syntax—for example, when creating tables or validating incoming data. This method translates a JSON schema into a SQL column type definition string, which is especially helpful for building integration scripts, automating ETL jobs, or generating documentation.

```python
pprint(converter.to_sql_string())
```

### ✅ Convert to Apache Iceberg Schema

You can now (as of version 1.2.0) convert a JSON Schema directly into an Iceberg-compatible schema using PyIceberg:

```python
from schemaworks.converter import JsonSchemaConverter

json_schema = {
    "type": "object",
    "properties": {
        "uid": {"type": "string"},
        "details": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "active": {"type": "boolean"}
            },
            "required": ["score"]
        }
    },
    "required": ["uid"]
}

converter = JsonSchemaConverter(json_schema)
iceberg_schema = converter.to_iceberg_schema()
```

### ✅ Handle decimals in JSON safely

Custom encoder to convert `Decimal` objects to `int` or `float` for JSON serialization.

This avoids serialization errors caused by unsupported Decimal types.
It does not preserve full precision—conversion uses built-in float or int types.

```python
from schemaworks.utils import DecimalEncoder
from decimal import Decimal
import json

data = {"price": Decimal("19.99")}
print(json.dumps(data, cls=DecimalEncoder))  # Output: {"price": 19.99}
```

### ✅ Validate data

```python
from schemaworks.validators import PythonTypeValidator

schema = {
    "type": "object",
    "properties": {
        "created_at": {"type": "datetime"},
        "price": {"type": "float"},
        "active": {"type": "bool"}
    }
}

data = {
    "created_at": "2023-01-01T00:00:00",
    "price": 10.5,
    "active": True
}

validator = PythonTypeValidator()
validator.validate(data, schema)
```

### ✅ Make data conform to schema

You can also use `.conform()` to enforce schema types and fill in missing values with sensible defaults:

```python
conformed_data = validator.conform(data, schema, fill_missing=True)
```

## 📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software, provided that you include the original copyright
notice and this permission notice in all copies or substantial portions of the software.

For full terms, see the [MIT license](https://opensource.org/license/mit).

## 🧑‍💻 Author

Anatol Jurenkow

Cloud Data Engineer | AWS Enthusiast | Iceberg Fan

[GitHub](https://github.com/anatol-ju) · [LinkedIn](https://de.linkedin.com/in/anatol-jurenkow)
