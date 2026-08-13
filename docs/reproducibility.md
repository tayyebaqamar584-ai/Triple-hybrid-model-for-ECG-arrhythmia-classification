# Reproducibility Checklist

1. Install dependencies in a fresh Fedora/Linux virtual environment.
2. Obtain the MIT-BIH Arrhythmia Database from its official source and record its
   version and citation.
3. Place the required records under the documented local data directory.
4. Run `python src/run_full_pipeline.py` from the repository root.
5. Record the source revision, Python version, dependency versions, random seed, split
   summary, and generated metric files.
6. Confirm that the reported experiment uses the documented beat-wise protocol.

The current pipeline is not a patient-independent evaluation. Do not compare its
metrics with inter-patient studies without resolving that methodological difference.