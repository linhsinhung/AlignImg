# AlignImg 1.10 final RF/MRA validation

This validation freezes one production path before any final real-data run:

- `candidate_scoring="fourier"`
- `reference_update="fourier"`
- `score_model="fourier_ncc"`
- CUDA batch size 512 with an 80% memory limit

Empirical whitening remains opt-in and is not part of this baseline. The goal is
to validate the complete alignment workflow, not to tune AlignImg until its
partition reproduces RELION 2D classification.

## Gate 1: 1000-particle workflow

Run the complete RF, feedback, and external comparison workflow with:

```bash
python tools/re2dc_70s_final_validation.py \
  --backend cuda \
  --batch-size 512 \
  --memory-fraction 0.8 \
  --output validation-results/re2dc-70s-final-n1000.json
```

The runner performs three sequential phases:

1. K=10 reference-free alignment for 15 iterations, using 10 annealing rounds
   and seed 0.
2. Five iterations each of fixed feedback and corrective feedback. Fixed
   feedback uses hard assignments with trust 1.0. Corrective feedback uses RF
   responsibilities with trust 0.9.
3. Label-permutation-invariant comparison of RF, fixed, and corrective
   assignments against the RELION partition.

Technical pass/fail checks are deliberately narrow:

- every required output array and reference is finite;
- every responsibility row is normalized within `2e-5`;
- result and reference shapes match the input and K;
- fixed feedback preserves every RF hard assignment;
- every phase uses the frozen Fourier/uniform-NCC production path;
- all expected artifacts and reports are present.

ARI, NMI, label agreement, FRC, occupancy, and corrective reassignment are
recorded as scientific observations. They do not determine whether this gate
passes, and RELION assignments are not treated as biological ground truth.

## Gate 2: 5000 particles and three seeds

Run this only after Gate 1 passes:

```bash
python tools/re2dc_70s_rf_validation.py \
  --stack data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.mrcs \
  --backend cuda \
  --components 10 \
  --iterations 20 \
  --anneal-iterations 10 \
  --seeds 0 1 2 \
  --angle-samples 128 \
  --translation-range 4 \
  --candidate-scoring fourier \
  --score-model fourier_ncc \
  --reference-update fourier \
  --batch-size 512 \
  --memory-fraction 0.8 \
  --output validation-results/re2dc-70s-final-n5000-k10-i20.json
```

Then compare the three final assignment sets with RELION:

```bash
python tools/re2dc_70s_relion_benchmark.py \
  --prepared-star data/re2dc_70s_testdata/prepared/re2dc_70s_n5000_s128.star \
  --rf-results \
    validation-results/re2dc-70s-final-n5000-k10-i20.seed-0.result.npz \
    validation-results/re2dc-70s-final-n5000-k10-i20.seed-1.result.npz \
    validation-results/re2dc-70s-final-n5000-k10-i20.seed-2.result.npz \
  --output validation-results/re2dc-70s-final-n5000-k10-i20.relion.json
```

This is the single formal baseline. If it exposes a clear failure, investigate
one factor at a time. Do not begin a parameter sweep merely to improve agreement
with the RELION classification result.

