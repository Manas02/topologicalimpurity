# Topological Impurity measure for Decision Trees and Random Forests for QSAR

Packaging and versioning -- let Poetry do the work.

Install poetry -
```sh
pipx install poetry
```

Create venv and install packages using poetry
```
cd topoinfogain/
poetry install
```

1. Build .so file from cython
```sh
cd src
python setup.py build_ext --inplace
```

2. Run benchmark
```sh
python benchmark.py
```

results are stored in `metrics_results.csv`.

3. Test on one dataset

```sh
python benchmark_one.py
```

- `tdt.py` -- pure python implementation
- `tree.py` -- cython implmentation

