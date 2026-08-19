# Methods

## Data and preprocessing

I read the Figshare `.mat` files (HDF5/v7.3) with h5py, extracting the image, the tumour label, and
the patient ID from each `cjdata` record. I exclude the indices the dataset documents as corrupt.
Every image is converted to 8-bit with a single documented per-image min-max rule and written as a
PNG, alongside a manifest that records the source, label, and patient ID for every slice. The
SARTAJ JPEGs (used only for the audit) are read the same way, with the non-tumour folder ignored.

## Splits

Two splitting strategies: a random split over slices, and a patient-grouped split that keeps all of
a patient's slices on one side. Both are seeded and reproducible; the grouped split asserts that no
patient appears in both train and test.

## Model and training

EfficientNet-B0 (timm), initialised from ImageNet weights, fine-tuned with Adam and cross-entropy.
Grayscale MRIs are replicated to three channels and resized. Checkpoints are written each epoch and
training can resume from the latest.

## Hyperparameters

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B0 (timm), ImageNet-pretrained |
| Optimiser | Adam |
| Learning rate | 3e-4 |
| Epochs | 5 |
| Batch size | 32 |
| Image size | 224 x 224 |
| Channels | grayscale replicated to 3 |
| Normalisation | ImageNet mean/std |
| Train augmentation | resize, random horizontal flip |
| Test transform | resize, normalise |
| Split | patient-grouped or random, test fraction 0.2 |
| Seeds | 0-9 for R1, 0 for R2-R5 |
| Loss | cross-entropy |

## Experiments

- **R1, leakage.** Train under each split strategy across three seeds and compare test accuracy.
- **Audit.** Perceptual-hash (pHash) every prepared image and, for each SARTAJ image, find its
  smallest Hamming distance to any Figshare image. Any pair within a small threshold is a
  cross-source near-duplicate; a single such pair fails the independence check.
- **R2, acquisition shift.** Apply six controlled corruptions (brightness, contrast, Gaussian
  noise, blur, downsampling, and an MRI bias field) at five severities to the held-out test scans
  and measure accuracy and calibration as a function of severity.
- **R3, diagnosis.** Grad-CAM heatmaps of clean vs shifted scans; a 2D embedding of the model's
  features; and a domain classifier trained to separate clean from shifted features (its AUC
  quantifies how visible the shift is inside the model).
- **R4, mitigation.** Compare the baseline against a training-matched intensity normalisation, a
  naive histogram equalisation, and AdaBN (recomputing batch-norm statistics on the shifted scans,
  without labels), reporting external accuracy, external calibration, and clean accuracy.
- **R5, uncertainty.** Fit a temperature on clean (in-distribution) logits, apply it to shifted
  scans, and report calibration before and after. Then use the top softmax probability as a
  confidence score to refer the least-confident scans away, and report accuracy versus coverage.

## Calibration metric

Expected Calibration Error is computed with a fixed number of bins throughout, so calibration
numbers are comparable across experiments.

## Reproducibility

The Figshare set has 3,064 slices from 233 patients. Fifteen slices documented as corrupt are excluded (indices 954-956, 1069-1075, and 1202-1206), leaving 3,049 usable slices across all 233 patients. `prepare_data.py` asserts these counts (3,064 slices read and 233 unique patient IDs) and fails loudly if they do not hold, so a mismatched dataset version is caught immediately.
