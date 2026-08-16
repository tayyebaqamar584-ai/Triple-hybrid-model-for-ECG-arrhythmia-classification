# Pre-Publication Paper Writing Checklist

## Item 2: Beat-Wise Split Protocol (Methods Section)

**Template paragraph to add to the Methods section:**

```
Data Split Protocol and Evaluation Strategy

The dataset was split at the beat level using stratified random sampling into training (75%), 
validation (15%), and test (10%) sets, based on the per-beat class distribution. This is an 
intra-patient evaluation protocol: beats from the same patient record may appear across multiple 
splits. This approach reflects a real-world scenario where a pre-trained classifier must identify 
arrhythmias in previously unseen beats from a patient whose baseline rhythm was seen during training. 
Importantly, this evaluation protocol differs from inter-patient protocols (such as AAMI EC57, 
de Chazal et al. 2004) where entire patient records are held out as test sets. Results from this 
beat-level evaluation should not be directly compared to inter-patient studies without accounting 
for this methodological difference. For reference, we retained the record-level split logic for 
potential inter-patient evaluation, but the primary results reported here use the intra-patient, 
beat-wise split.
```

**Optional secondary result for comparison:**
If reviewers/readers expect inter-patient comparison, consider adding a secondary results table:

```
Inter-Patient Evaluation (Record-Level Split)

For comparison with inter-patient protocols, we evaluated the same pipeline using a record-level 
split where entire patient records are reserved for test (following AAMI EC57 conventions). 
Results were [X% accuracy, Y% F1-macro], which are lower than the intra-patient results, 
confirming that patient-level generalization remains a more challenging problem.
```

---

## Item 3: Fiducial Point Detection (Methods Section)

**Template paragraph to add to the Methods section (under Feature Extraction):**

```
Fiducial Point Detection and Interval Measurement

Automated fiducial points (QRS onset/offset, P-wave and T-wave extents, PR and QT intervals) 
were estimated using a lightweight heuristic algorithm based on threshold-crossing scans from 
the R-peak, rather than a validated clinical delineation algorithm (e.g., Pan-Tompkins, 
wavelet-based delineators). These estimates serve as engineered features for the classifier 
and should not be interpreted as clinically calibrated interval measurements. The heuristic 
nature of this approach is a simplification made for computational efficiency; for clinical 
applications requiring precise interval measurements, validated delineation methods are 
recommended.
```

---

## Checklist: When to update these sections

- [ ] Beat-wise paragraph → add to Methods section, under "Data and Splits"
- [ ] Fiducial detection caveat → add to Methods section, under "Feature Extraction"
- [ ] Update abstract or introduction if beat-wise protocol is a key contribution
- [ ] Decide if inter-patient table should be included as secondary result
- [ ] Cite de Chazal et al. (2004) and AAMI EC57 standard if discussing inter-patient protocols
- [ ] Review limitations section to ensure all methodology caveats are noted
- [ ] Have co-authors review these additions before submission
