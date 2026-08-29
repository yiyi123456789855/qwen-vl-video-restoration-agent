# Held-out Benchmark Report

## Primary metrics

| Metric | Raw VLM | Fused routing |
|---|---:|---:|
| Overall tool accuracy | 48.1% | 55.6% |
| ID route accuracy | 72.2% | 50.0% |
| OOD rejection rate | 0.0% | 66.7% |
| Clean passthrough rate | 66.7% | 66.7% |
| Clean false activation rate | 33.3% | 0.0% |
| Manual review rate | 3.7% | 51.9% |

## Accuracy by generation

| Generation | Runs | Raw accuracy | Fused accuracy |
|---|---:|---:|---:|
| blur | 3 | 66.7% | 66.7% |
| clean | 3 | 66.7% | 66.7% |
| jpeg_quality10 | 3 | 0.0% | 33.3% |
| lowlight_gain012 | 3 | 100.0% | 100.0% |
| lowlight_gain030 | 3 | 100.0% | 0.0% |
| mixed_blur_noise | 3 | 0.0% | 66.7% |
| noise_sigma12 | 3 | 33.3% | 0.0% |
| noise_sigma35 | 3 | 66.7% | 66.7% |
| unknown_color_cast | 3 | 0.0% | 100.0% |

## Decision sources

| Source | Runs |
|---|---:|
| abstain_objective_uncertain | 4 |
| abstain_vlm_objective_disagreement | 10 |
| vlm_objective_agreement | 13 |

## Per-case results

| Case | Generation | OOD | Expected | Raw | Fused | Raw correct | Fused correct | Source | Confidence |
|---|---|---:|---|---|---|---:|---:|---|---:|
| GOPR0396_11_00__clean | clean | no | none | none | none | yes | yes | vlm_objective_agreement | 0.705 |
| GOPR0396_11_00__blur | blur | no | deblur | none | none | no | no | vlm_objective_agreement | 0.833 |
| GOPR0396_11_00__noise_sigma12 | noise_sigma12 | no | denoise | none | manual_review | no | no | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0396_11_00__noise_sigma35 | noise_sigma35 | no | denoise | denoise | denoise | yes | yes | vlm_objective_agreement | 1.000 |
| GOPR0396_11_00__lowlight_gain030 | lowlight_gain030 | no | enhance_lowlight | enhance_lowlight | manual_review | yes | no | abstain_objective_uncertain | 0.000 |
| GOPR0396_11_00__lowlight_gain012 | lowlight_gain012 | no | enhance_lowlight | enhance_lowlight | enhance_lowlight | yes | yes | vlm_objective_agreement | 0.820 |
| GOPR0396_11_00__jpeg_quality10 | jpeg_quality10 | yes | manual_review | none | none | no | no | vlm_objective_agreement | 0.931 |
| GOPR0396_11_00__mixed_blur_noise | mixed_blur_noise | yes | manual_review | denoise | denoise | no | no | vlm_objective_agreement | 0.851 |
| GOPR0396_11_00__unknown_color_cast | unknown_color_cast | yes | manual_review | denoise | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0868_11_00__clean | clean | no | none | none | none | yes | yes | vlm_objective_agreement | 0.790 |
| GOPR0868_11_00__blur | blur | no | deblur | deblur | deblur | yes | yes | vlm_objective_agreement | 0.677 |
| GOPR0868_11_00__noise_sigma12 | noise_sigma12 | no | denoise | denoise | manual_review | yes | no | abstain_objective_uncertain | 0.648 |
| GOPR0868_11_00__noise_sigma35 | noise_sigma35 | no | denoise | denoise | denoise | yes | yes | vlm_objective_agreement | 1.000 |
| GOPR0868_11_00__lowlight_gain030 | lowlight_gain030 | no | enhance_lowlight | enhance_lowlight | manual_review | yes | no | abstain_objective_uncertain | 0.000 |
| GOPR0868_11_00__lowlight_gain012 | lowlight_gain012 | no | enhance_lowlight | enhance_lowlight | enhance_lowlight | yes | yes | vlm_objective_agreement | 0.822 |
| GOPR0868_11_00__jpeg_quality10 | jpeg_quality10 | yes | manual_review | denoise | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0868_11_00__mixed_blur_noise | mixed_blur_noise | yes | manual_review | deblur | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0868_11_00__unknown_color_cast | unknown_color_cast | yes | manual_review | denoise | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0871_11_00__clean | clean | no | none | denoise | manual_review | no | no | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0871_11_00__blur | blur | no | deblur | deblur | deblur | yes | yes | vlm_objective_agreement | 0.893 |
| GOPR0871_11_00__noise_sigma12 | noise_sigma12 | no | denoise | deblur | manual_review | no | no | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0871_11_00__noise_sigma35 | noise_sigma35 | no | denoise | manual_review | manual_review | no | no | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0871_11_00__lowlight_gain030 | lowlight_gain030 | no | enhance_lowlight | enhance_lowlight | manual_review | yes | no | abstain_objective_uncertain | 0.000 |
| GOPR0871_11_00__lowlight_gain012 | lowlight_gain012 | no | enhance_lowlight | enhance_lowlight | enhance_lowlight | yes | yes | vlm_objective_agreement | 0.826 |
| GOPR0871_11_00__jpeg_quality10 | jpeg_quality10 | yes | manual_review | none | none | no | no | vlm_objective_agreement | 0.901 |
| GOPR0871_11_00__mixed_blur_noise | mixed_blur_noise | yes | manual_review | deblur | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
| GOPR0871_11_00__unknown_color_cast | unknown_color_cast | yes | manual_review | denoise | manual_review | no | yes | abstain_vlm_objective_disagreement | 0.000 |
