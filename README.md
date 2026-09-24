# dv-ferm-analytics

Reusable Python tools for industrial analytics, statistical analysis, benchmarking, value-tree calculations, visualization, and Microsoft Word report generation.

## Requirements

- Python `>=3.13,<3.14`

The main dependencies are declared in `pyproject.toml` and include pandas, SciPy, scikit-learn, SHAP, Matplotlib, python-docx, lxml, requests, and NetworkX.

## Installation

After publication on PyPI:

```bash
pip install dv-ferm-analytics
```

For local development with Poetry:

```bash
poetry install
poetry run pytest
```

## Main modules

The package currently provides reusable components for:

- Top-N benchmarking of industrial units;
- value-tree analysis against a benchmark;
- value-tree comparison with the best historical year;
- perfect-year theoretical reference analysis;
- comparison against selected reference units;
- comparison with the same unit in the previous year;
- RTC conference and reconciliation;
- indirect RTC/RGD estimation;
- multiple linear regression for RTC;
- Pearson correlation and simple linear regression;
- year-over-year comparative tables with paired significance tests;
- CEPEA sugar and ethanol price retrieval;
- reusable plotting helpers;
- Microsoft Word technical report generation.

Detailed usage instructions are available in:

```text
docs/MANUAL_MODULOS_DV_FERM_ANALYTICS.md
```

## Quick example

```python
from dv_ferm_analytics.analysis import calculate_top_benchmark

result = calculate_top_benchmark(
    df,
    autonoma=False,
    difusor=False,
    top_n=5,
)

print(result.top_units)
print(result.indicator_means)
```

For specialized modules, import directly from the corresponding module, for example:

```python
from dv_ferm_analytics.analysis.comparative_tables import (
    ComparisonIndicator,
    calculate_year_comparison,
)
```

## CEPEA access

`dv_ferm_analytics.market.get_cepea_prices()` performs an HTTP request to public CEPEA pages. Internet access is therefore required only when this functionality is used.

## Development checks

Before creating a release:

```bash
poetry check
poetry run pytest
poetry build
```

Inspect the generated artifacts before publishing:

```bash
python -m zipfile -l dist/dv_ferm_analytics-<version>-py3-none-any.whl
```

## Data privacy and security

This repository must contain only generic, reusable source code and synthetic/anonymized test data.

Do not commit or distribute:

- real client or plant datasets;
- client or plant names used in private projects;
- credentials, tokens, passwords, API keys, or database connection strings;
- private price configurations;
- corporate templates or private reports;
- local virtual environments, IDE metadata, generated outputs, or local filesystem paths.

The `.gitignore` is configured to exclude common local, private, and generated artifacts. Always review `git status` and the built wheel/sdist before publishing.

## Authors

Developed by **dv-analytics**.

## License

This project is distributed under a **proprietary/restricted license**.
Public availability of the source code or package does not grant permission to
reuse, modify, redistribute, sublicense, commercialize, or create derivative
works except when expressly authorized in writing by the copyright holder.

See [`LICENSE`](LICENSE) for the applicable terms.
