# Exploratory distillation: detailed numerical record

The manuscript reports the interpretation concisely; this record preserves the detailed values and measurement limits. LaTeX notation follows the source manuscript.

In the exploratory development population, the distilled frozen-small-pLM student reached
ROC-AUC 0.75968, compared with B1 at 0.71159, the independently tuned same-encoder hard-only
head at 0.75138 and the teacher at 0.77993 (Figure~\ref{fig:distill}a).
Distilled minus B1 was +0.04809 (conditional 95\% CI [0.04063, 0.05548]); distilled minus
hard-only was +0.00830 [0.00237, 0.01418]. The matched-$C$ hard-only head reached 0.74996,
with a paired distillation gain of +0.00972 [0.00669, 0.01287]. All outer folds selected
$\alpha=0.5$ and $C=10$ for distillation; hard-only selected $C=10^{-3}$.

The student retained 70.4\% of the teacher's gain above B1. This must not all be credited to distillation; it includes the small encoder's
pretraining contribution: hard-only retained 58.2\%, and distillation added approximately
12.1 percentage points, recovering 29.1\% of the remaining teacher--hard-student gap.
Distilled log loss/Brier were 0.59318/0.20306, better than B1's 0.61951/0.21636 but slightly
worse than hard-only's 0.58940/0.20235. Thus the ranking improvement did not also improve
probability losses relative to the tuned hard-only control.

The complete predictor has 33,762,666 learned parameters, about 35.5 times fewer than the
approximately 1.2-billion-parameter teacher. Its FP32 encoder and head occupy 135,092,558 bytes
(135.1 MB); unused pooling and contact heads were removed. A synthetic 500-residue input took
median 242.97 ms end-to-end on one CPU thread (three warmups, ten repeats), versus 0.006 ms
for the cached-feature head alone. NumPy-head predictions matched all 8,198 stored out-of-fold
scores; fresh raw-sequence checks covered four probes. FP16 storage reduced the encoder file
to 67,546,600 bytes, but predictive equivalence was not established.

Earlier compact models did not show the same transfer. The fixed TriPepLite audit produced
hard-only/distilled AUCs of 0.641/0.652 against B1/teacher values of 0.707/0.790;
the paired distillation interval included zero. The independent post-hoc reconstruction
likewise remained below B1 (Supplementary Method 16 and Table S6).
Later linear, radial-basis and neural students reached 0.72720, 0.72791 and 0.72877,
but none established a gain over the independently tuned B1/context blend at 0.72772
(Figure~\ref{fig:distill}b). The linear export was 58,276 bytes and took 0.513 ms per
synthetic 500-residue input. Its small size alone did not establish useful teacher transfer.

No student was promoted under the original criteria. The small-pLM student retained useful
additional discrimination but remained below the 80\% retention target and above the 5 MiB
and 1 ms limits. It requires ESM/PyTorch and is teacher-free at inference, not language-model-free.
Teacher latency, smartphone performance, peak process memory and energy were not measured.

