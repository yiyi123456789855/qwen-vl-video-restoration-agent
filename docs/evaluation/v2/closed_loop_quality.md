# Closed-Loop Restoration Quality Report

Included runs: 8. Excluded forced/stress runs: 0.

## Per-run results

| Case | Tool | Expected | OOD | Degradation | Final status | Attempts | Attempt states | Quality | Tool s | Total s | Peak GB |
| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| GOPR0868_11_00__blur | deblur | deblur | no | blur | accept | 1 | accept | 1.0000 | 8.440 | 15.257 | - |
| GOPR0871_11_00__blur | deblur | deblur | no | blur | accept | 1 | accept | 1.0000 | 8.437 | 15.338 | - |
| GOPR0396_11_00__mixed_blur_noise | denoise | manual_review | yes | noise | accept | 1 | accept | 1.0000 | 1.460 | 8.766 | - |
| GOPR0396_11_00__noise_sigma35 | denoise | denoise | no | noise | manual_review | 1 | manual_review | 0.6667 | 1.385 | 8.685 | - |
| GOPR0868_11_00__noise_sigma35 | denoise | denoise | no | noise | manual_review | 1 | manual_review | 0.6667 | 1.508 | 8.772 | - |
| GOPR0396_11_00__lowlight_gain012 | enhance_lowlight | enhance_lowlight | no | low_light | accept | 1 | accept | 1.0000 | 3.180 | 10.081 | 2.105 |
| GOPR0868_11_00__lowlight_gain012 | enhance_lowlight | enhance_lowlight | no | low_light | accept | 1 | accept | 1.0000 | 3.107 | 9.935 | 2.105 |
| GOPR0871_11_00__lowlight_gain012 | enhance_lowlight | enhance_lowlight | no | low_light | accept | 1 | accept | 1.0000 | 3.210 | 10.076 | 2.105 |

## Aggregate by tool

| Tool | Runs | Accept | Retry | Manual review | Stop | Mean attempts | Mean quality | Mean tool s | Mean total s | Mean peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 8 | 75.0% | 0.0% | 25.0% | 0.0% | 1.000 | 0.9167 | 3.841 | 10.864 | 2.105 |
| deblur | 2 | 100.0% | 0.0% | 0.0% | 0.0% | 1.000 | 1.0000 | 8.438 | 15.297 | - |
| denoise | 3 | 33.3% | 0.0% | 66.7% | 0.0% | 1.000 | 0.7778 | 1.451 | 8.741 | - |
| enhance_lowlight | 3 | 100.0% | 0.0% | 0.0% | 0.0% | 1.000 | 1.0000 | 3.166 | 10.031 | 2.105 |

## Metric definitions

- Accept rate: final closed-loop status is `accept`.
- Retry rate: more than one restoration attempt was executed.
- Manual review rate: final status is `manual_review`.
- Stop rate: severe quality harm triggered the safety stop.
- Pressure tests and forced-tool runs are excluded by default.
