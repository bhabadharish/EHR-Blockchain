.PHONY: all install test test-crypto test-fhir benchmarks evaluate dashboard clean

PYTHON = python3

all: test benchmarks evaluate

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-crypto:
	pytest tests/test_crypto.py tests/test_integrity.py -v

test-fhir:
	pytest tests/test_fhir_validation.py tests/test_api.py -v

benchmarks:
	$(PYTHON) scripts/run_crypto_benchmarks.py
	$(PYTHON) scripts/run_blockchain_benchmarks.py
	$(PYTHON) scripts/run_scalability.py
	$(PYTHON) scripts/run_attack_simulation.py

evaluate:
	$(PYTHON) scripts/evaluate_models.py
	$(PYTHON) scripts/generate_tables.py
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/check_consistency.py

dashboard:
	streamlit run dashboard/streamlit_app.py

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
