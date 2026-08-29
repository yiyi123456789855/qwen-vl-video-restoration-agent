# V2 Frozen Held-out Evaluation

## Evaluation protocol

- Frozen code and benchmark tag: `v2.0.0-rc1`
- Frozen commit: `bbe90c02f798a18c4487260863b7e8e2c9e63c66`
- Benchmark: `gopro_heldout_v2`
- Split: `final_test`
- Scenes: `GOPR0396_11_00`, `GOPR0868_11_00`, `GOPR0871_11_00`
- Cases: 27 cases, 5 ordered frames per case
- Composition: 18 in-distribution (ID) cases and 9 out-of-distribution (OOD) cases
- Diagnosis mode: `single`
- Routing policy: `agreement_only`
- Maximum restoration attempts: 2

The code, prompts, objective-prior thresholds, quality thresholds, retry policy, and benchmark split were frozen before this evaluation. No parameter was tuned using the results below.

## Execution summary

| Metric | Result |
|---|---:|
| Requested cases | 27 |
| Completed cases | 27 |
| Failed runs | 0 |
| Final automatic acceptance | 11/27 (40.7%) |
| Final manual review | 16/27 (59.3%) |
| Total evaluation time | 222.96 s |

## Routing results

| Metric | Raw VLM | Fused routing |
|---|---:|---:|
| Overall exact tool accuracy | 48.1% | 55.6% |
| ID exact route accuracy | 72.2% | 50.0% |
| OOD rejection rate | 0.0% | 66.7% |
| Clean passthrough rate | 66.7% | 66.7% |
| Clean false activation rate | 33.3% | 0.0% |
| Manual review rate | 3.7% | 51.9% |

The agreement-only policy trades coverage for safety. Among the 18 ID cases, it automatically routed 10 cases (55.6% ID coverage), and 9 of those 10 accepted routes were correct (90.0% accepted-ID route accuracy). However, eight ID cases were sent to manual review, so exact ID route accuracy decreased to 50.0% when abstentions were counted as incorrect.

Among nine OOD cases, six were correctly rejected and three were incorrectly accepted, giving a 66.7% OOD rejection rate. The fused policy eliminated automatic restoration on clean inputs, reducing the clean false activation rate from 33.3% to 0.0%.

## Closed-loop restoration quality

Eight cases entered a restoration tool. Six passed the objective quality gate and two were redirected to manual review.

| Tool | Executed | Accepted | Manual review | Acceptance rate | Mean tool time |
|---|---:|---:|---:|---:|---:|
| Deblur | 2 | 2 | 0 | 100.0% | 8.438 s |
| Denoise | 3 | 1 | 2 | 33.3% | 1.451 s |
| Low-light enhancement | 3 | 3 | 0 | 100.0% | 3.166 s |
| All tools | 8 | 6 | 2 | 75.0% | 3.841 s |

The two rejected denoising outputs failed the gradient-retention requirement, indicating excessive smoothing. The failure-aware retry policy did not repeat an ineffective overlap-only retry and instead transferred these cases directly to manual review.

## Observed failure modes

1. One `GOPR0396_11_00` blur case was incorrectly passed through because both the VLM and objective prior selected `none`.
2. Mild noise (`noise_sigma12`) was frequently rejected, showing that the current conservative policy has limited coverage for subtle degradation.
3. Mild low-light inputs (`lowlight_gain030`) were routed to manual review because the objective prior was uncertain despite correct VLM predictions.
4. One mixed blur-noise OOD case was incorrectly routed to denoising and passed the output-quality gate. This demonstrates that output-quality metrics cannot replace degradation/OOD recognition.
5. JPEG OOD detection was inconsistent: one of three JPEG cases was rejected, while two were incorrectly passed through as `none`.

## Interpretation

The main result is not maximum closed-set classification accuracy. The system demonstrates a reproducible, auditable pipeline on one RTX 4090 that combines multimodal diagnosis, objective-prior routing, abstention, multiple restoration tools, and post-restoration quality control.

On the frozen held-out split, fusion improved overall exact routing accuracy from 48.1% to 55.6%, increased OOD rejection from 0.0% to 66.7%, and reduced clean false activation from 33.3% to 0.0%. This safety improvement came with lower automatic ID coverage and a 51.9% routing-stage manual-review rate. The quality loop independently blocked two over-smoothed denoising outputs, but it did not detect one semantically incorrect OOD route.

These results support the value of agreement-based abstention and post-restoration quality checks while clearly identifying calibration, mild-degradation recognition, and mixed-degradation handling as future work.

## Reproducibility artifacts

- `benchmarks/heldout_v2.json`: frozen benchmark manifest
- `docs/evaluation/v2/run_summary_single.json`: execution summary
- `docs/evaluation/v2/routing_comparison.md`: routing report
- `docs/evaluation/v2/routing_comparison.csv`: per-case routing data
- `docs/evaluation/v2/routing_comparison.json`: structured routing metrics
- `docs/evaluation/v2/closed_loop_quality.md`: quality-control report
- `docs/evaluation/v2/closed_loop_quality.csv`: per-run quality data
- `docs/evaluation/v2/closed_loop_quality.json`: structured quality metrics
