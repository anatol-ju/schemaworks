# SchemaWorks

**SchemaWorks** is a Python library for converting between different schema definitions, such as JSON Schema, Spark DataTypes, SQL type strings, and more. It aims to simplify working with structured data across multiple data engineering and analytics platforms.

## 🔧 Features

- Convert JSON Schema to:
  - Apache Spark StructType
  - SQL column type strings
  - Python dtypes dictionaries
- Convert Spark schemas and dtypes to JSON Schema
- Generate JSON Schemas from example data
- Flatten nested schemas for easier inspection or mapping
- Utilities for handling Decimal encoding and schema inference

## 🚀 Use Cases

- Building pipelines that consume or produce data in multiple formats
- Ensuring schema consistency across Spark, SQL, and data validation layers
- Automating schema generation from sample data for prototyping
- Simplifying developer tooling with schema introspection

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

converter = JsonSchemaConverter(json_schema=schema)

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

## 🧪 Testing

Run unit tests using pytest:
```bash
poetry run pytest
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

You are free to use, modify, and distribute this software under the same license, provided that:
- You include the original license and copyright notice.
- You disclose source code when distributing your modified version.
- You do not impose additional restrictions beyond those of the GPL.

Commercial use is permitted, but your derivative work must also be open source under GPLv3.

For full terms, see the [GNU GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).

## 🧑‍💻 Author

Anatol Jurenkow

Cloud Data Engineer | AWS Enthusiast | Iceberg Fan

[GitHub](https://github.com/anatol-ju) · [LinkedIn](https://de.linkedin.com/in/anatol-jurenkow)
