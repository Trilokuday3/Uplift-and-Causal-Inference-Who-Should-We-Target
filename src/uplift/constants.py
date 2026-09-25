FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
EXPOSURE_COL = "exposure"
VISIT_COL = "visit"
CONVERSION_COL = "conversion"
# Confirmed against the real Criteo v2.1 dataset (11,882,655 / 13,979,592 = 0.8500),
# not the source doc's approximate 0.846 - the SRM check correctly caught the mismatch
# on the first real-data run.
EXPECTED_TREATMENT_RATE = 0.85
