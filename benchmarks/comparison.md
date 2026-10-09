Before: `benchmarks/results_before.json` (ext sha 198533b8edd67de0); After: `benchmarks/results_after.json` (ext sha ecf20dca42257324); GPU NVIDIA GeForce RTX 4090, torch 2.7.1+cu118, float32.

Time = median CUDA-event ms per iteration (GPU-side span of one iteration incl. launch gaps); Mem = peak allocator increment over the pre-iteration baseline (MiB). optimizer.step not executed in these rows.

fwd_bwd_reg = separate calls; fwd_bwd_reg_fused = joint forward_with_regularization. Both use the original separate API for the before column. No fastest-of-two selection.

| config | after scenario | backend | before ms | after ms | speedup | before MiB | after MiB | actual backend (after) |
|---|---|---|---:|---:|---:|---:|---:|---|
| L-B1-16-G5 | infer | gpu_ref | 0.271 | 0.281 | 0.97x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | infer | gpu_ref_compile | 0.075 | 0.080 | 0.94x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | infer | gpu_native | 0.037 | 0.042 | 0.90x | 0.0 | 0.0 | cuda |
| L-B1-16-G5 | fwd_bwd | gpu_ref | 0.527 | 0.551 | 0.96x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd | gpu_ref_compile | 0.171 | 0.168 | 1.01x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd | gpu_native | 0.179 | 0.135 | 1.32x | 0.0 | 0.0 | cuda |
| L-B1-16-G5 | fwd_bwd_reg | gpu_ref | 1.228 | 1.334 | 0.92x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd_reg | gpu_ref_compile | 0.220 | 0.315 | 0.70x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd_reg | gpu_native | 0.937 | 0.492 | 1.91x | 0.0 | 0.0 | cuda |
| L-B1-16-G5 | fwd_bwd_reg_fused | gpu_ref | 1.228 | 0.846 | 1.45x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd_reg_fused | gpu_ref_compile | 0.220 | 0.304 | 0.72x | 0.0 | 0.0 | reference |
| L-B1-16-G5 | fwd_bwd_reg_fused | gpu_native | 0.937 | 0.383 | 2.45x | 0.0 | 0.0 | cuda |
| L-B256-64-G5 | infer | gpu_ref | 0.380 | 0.381 | 1.00x | 3.9 | 3.9 | reference |
| L-B256-64-G5 | infer | gpu_ref_compile | 0.074 | 0.076 | 0.98x | 1.4 | 1.4 | reference |
| L-B256-64-G5 | infer | gpu_native | 0.037 | 0.042 | 0.89x | 0.9 | 0.9 | cuda |
| L-B256-64-G5 | fwd_bwd | gpu_ref | 0.535 | 0.604 | 0.89x | 4.0 | 4.0 | reference |
| L-B256-64-G5 | fwd_bwd | gpu_ref_compile | 0.185 | 0.228 | 0.81x | 1.4 | 1.4 | reference |
| L-B256-64-G5 | fwd_bwd | gpu_native | 0.194 | 0.201 | 0.96x | 2.9 | 1.2 | cuda |
| L-B256-64-G5 | fwd_bwd_reg | gpu_ref | 1.273 | 1.283 | 0.99x | 17.4 | 17.2 | reference |
| L-B256-64-G5 | fwd_bwd_reg | gpu_ref_compile | 0.251 | 0.358 | 0.70x | 8.9 | 8.9 | reference |
| L-B256-64-G5 | fwd_bwd_reg | gpu_native | 0.967 | 0.436 | 2.22x | 17.3 | 1.4 | cuda |
| L-B256-64-G5 | fwd_bwd_reg_fused | gpu_ref | 1.273 | 0.963 | 1.32x | 17.4 | 16.6 | reference |
| L-B256-64-G5 | fwd_bwd_reg_fused | gpu_ref_compile | 0.251 | 0.256 | 0.98x | 8.9 | 8.9 | reference |
| L-B256-64-G5 | fwd_bwd_reg_fused | gpu_native | 0.967 | 0.393 | 2.46x | 17.3 | 0.9 | cuda |
| L-B256-64-G5 | grid_update | gpu_ref | 5.939 | 5.976 | 0.99x | 9.2 | 9.0 | reference |
| L-B256-64-G5 | grid_update | gpu_native | 5.929 | 6.046 | 0.98x | 9.2 | 9.0 | cuda |
| L-B1024-256-G5 | infer | gpu_ref | 0.417 | 0.422 | 0.99x | 63.0 | 63.0 | reference |
| L-B1024-256-G5 | infer | gpu_ref_compile | 0.107 | 0.109 | 0.99x | 23.0 | 23.0 | reference |
| L-B1024-256-G5 | infer | gpu_native | 0.132 | 0.083 | 1.58x | 14.0 | 14.0 | cuda |
| L-B1024-256-G5 | fwd_bwd | gpu_ref | 0.593 | 0.609 | 0.97x | 64.0 | 64.0 | reference |
| L-B1024-256-G5 | fwd_bwd | gpu_ref_compile | 0.188 | 0.229 | 0.82x | 23.0 | 23.0 | reference |
| L-B1024-256-G5 | fwd_bwd | gpu_native | 0.345 | 0.159 | 2.17x | 46.5 | 18.5 | cuda |
| L-B1024-256-G5 | fwd_bwd_reg | gpu_ref | 9.636 | 9.952 | 0.97x | 1142.3 | 1043.0 | reference |
| L-B1024-256-G5 | fwd_bwd_reg | gpu_ref_compile | 3.972 | 4.077 | 0.97x | 526.0 | 526.0 | reference |
| L-B1024-256-G5 | fwd_bwd_reg | gpu_native | 9.293 | 0.638 | 14.56x | 1141.3 | 22.5 | cuda |
| L-B1024-256-G5 | fwd_bwd_reg_fused | gpu_ref | 9.636 | 9.422 | 1.02x | 1142.3 | 1034.0 | reference |
| L-B1024-256-G5 | fwd_bwd_reg_fused | gpu_ref_compile | 3.972 | 4.063 | 0.98x | 526.0 | 526.0 | reference |
| L-B1024-256-G5 | fwd_bwd_reg_fused | gpu_native | 9.293 | 0.599 | 15.51x | 1141.3 | 15.0 | cuda |
| L-B1024-256-G5 | grid_update | gpu_ref | 370.424 | 368.931 | 1.00x | 531.0 | 528.0 | reference |
| L-B1024-256-G5 | grid_update | gpu_native | 370.777 | 369.748 | 1.00x | 531.0 | 528.0 | cuda |
| L-B256-64-G64 | infer | gpu_ref | 0.384 | 0.385 | 1.00x | 26.1 | 26.1 | reference |
| L-B256-64-G64 | infer | gpu_ref_compile | 0.078 | 0.078 | 1.00x | 8.8 | 8.8 | reference |
| L-B256-64-G64 | infer | gpu_native | 0.075 | 0.067 | 1.13x | 5.5 | 5.5 | cuda |
| L-B256-64-G64 | fwd_bwd | gpu_ref | 0.601 | 0.541 | 1.11x | 26.2 | 26.2 | reference |
| L-B256-64-G64 | fwd_bwd | gpu_ref_compile | 0.182 | 0.180 | 1.01x | 8.8 | 8.8 | reference |
| L-B256-64-G64 | fwd_bwd | gpu_native | 0.212 | 0.143 | 1.49x | 21.0 | 9.1 | cuda |
| L-B256-64-G64 | fwd_bwd_reg | gpu_ref | 1.284 | 1.296 | 0.99x | 34.5 | 30.5 | reference |
| L-B256-64-G64 | fwd_bwd_reg | gpu_ref_compile | 0.253 | 0.356 | 0.71x | 15.4 | 17.5 | reference |
| L-B256-64-G64 | fwd_bwd_reg | gpu_native | 0.980 | 0.439 | 2.23x | 34.5 | 11.2 | cuda |
| L-B256-64-G64 | fwd_bwd_reg_fused | gpu_ref | 1.284 | 0.862 | 1.49x | 34.5 | 26.1 | reference |
| L-B256-64-G64 | fwd_bwd_reg_fused | gpu_ref_compile | 0.253 | 0.270 | 0.94x | 15.4 | 15.4 | reference |
| L-B256-64-G64 | fwd_bwd_reg_fused | gpu_native | 0.980 | 0.407 | 2.41x | 34.5 | 7.0 | cuda |
| L-B256-64-G64 | grid_update | gpu_ref | 13.657 | 13.246 | 1.03x | 34.5 | 34.3 | reference |
| L-B256-64-G64 | grid_update | gpu_native | 13.643 | 13.268 | 1.03x | 34.5 | 34.3 | cuda |
| L-B4096-256-G5 | infer | gpu_ref | 2.233 | 2.239 | 1.00x | 252.0 | 252.0 | reference |
| L-B4096-256-G5 | infer | gpu_ref_compile | 0.365 | 0.361 | 1.01x | 92.0 | 92.0 | reference |
| L-B4096-256-G5 | infer | gpu_native | 0.441 | 0.195 | 2.26x | 50.0 | 50.0 | cuda |
| L-B4096-256-G5 | fwd_bwd | gpu_ref | 2.386 | 2.398 | 1.00x | 256.0 | 256.0 | reference |
| L-B4096-256-G5 | fwd_bwd | gpu_ref_compile | 0.534 | 0.526 | 1.01x | 92.0 | 92.0 | reference |
| L-B4096-256-G5 | fwd_bwd | gpu_native | 1.154 | 0.360 | 3.21x | 166.5 | 50.0 | cuda |
| L-B4096-256-G5 | fwd_bwd_reg | gpu_ref | 66.794 | 68.037 | 0.98x | 4306.3 | 4172.0 | reference |
| L-B4096-256-G5 | fwd_bwd_reg | gpu_ref_compile | 42.754 | 44.467 | 0.96x | 2092.0 | 2092.0 | reference |
| L-B4096-256-G5 | fwd_bwd_reg | gpu_native | 65.863 | 1.924 | 34.24x | 4302.3 | 79.3 | cuda |
| L-B4096-256-G5 | fwd_bwd_reg_fused | gpu_ref | 66.794 | 66.768 | 1.00x | 4306.3 | 4136.0 | reference |
| L-B4096-256-G5 | fwd_bwd_reg_fused | gpu_ref_compile | 42.754 | 43.214 | 0.99x | 2092.0 | 2092.0 | reference |
| L-B4096-256-G5 | fwd_bwd_reg_fused | gpu_native | 65.863 | 1.848 | 35.64x | 4302.3 | 51.0 | cuda |
| L-B4096-256-G5 | grid_update | gpu_ref | 1461.654 | 1436.440 | 1.02x | 2124.0 | 2112.0 | reference |
| L-B4096-256-G5 | grid_update | gpu_native | 1461.310 | 1458.585 | 1.00x | 2124.0 | 2112.0 | cuda |
| KAN-16-32-8-B256 | infer | gpu_ref | 0.755 | 0.752 | 1.00x | 2.0 | 2.0 | reference |
| KAN-16-32-8-B256 | infer | gpu_ref_compile | 0.123 | 0.125 | 0.98x | 0.7 | 0.7 | reference |
| KAN-16-32-8-B256 | infer | gpu_native | 0.065 | 0.074 | 0.88x | 0.3 | 0.3 | cuda |
| KAN-16-32-8-B256 | fwd_bwd | gpu_ref | 1.494 | 1.397 | 1.07x | 3.3 | 3.3 | reference |
| KAN-16-32-8-B256 | fwd_bwd | gpu_ref_compile | 0.373 | 0.379 | 0.98x | 2.2 | 2.2 | reference |
| KAN-16-32-8-B256 | fwd_bwd | gpu_native | 0.323 | 0.345 | 0.94x | 1.4 | 0.9 | cuda |
| KAN-16-32-8-B256 | fwd_bwd_reg | gpu_ref | 3.819 | 3.494 | 1.09x | 5.7 | 6.5 | reference |
| KAN-16-32-8-B256 | fwd_bwd_reg | gpu_ref_compile | 0.467 | 0.577 | 0.81x | 2.9 | 4.8 | reference |
| KAN-16-32-8-B256 | fwd_bwd_reg | gpu_native | 1.894 | 0.937 | 2.02x | 3.3 | 1.2 | cuda |
| KAN-16-32-8-B256 | fwd_bwd_reg_fused | gpu_ref | 3.819 | 2.069 | 1.85x | 5.7 | 4.0 | reference |
| KAN-16-32-8-B256 | fwd_bwd_reg_fused | gpu_ref_compile | 0.467 | 0.486 | 0.96x | 2.9 | 2.9 | reference |
| KAN-16-32-8-B256 | fwd_bwd_reg_fused | gpu_native | 1.894 | 0.760 | 2.49x | 3.3 | 1.0 | cuda |
| KAN-16-32-8-B256 | grid_update | gpu_ref | 5.296 | 4.886 | 1.08x | 2.6 | 2.5 | reference |
| KAN-16-32-8-B256 | grid_update | gpu_native | 4.440 | 4.192 | 1.06x | 2.6 | 2.5 | cuda |
| KAN-16-32-8-B4096 | infer | gpu_ref | 0.747 | 0.752 | 0.99x | 31.6 | 31.6 | reference |
| KAN-16-32-8-B4096 | infer | gpu_ref_compile | 0.126 | 0.124 | 1.02x | 11.6 | 11.6 | reference |
| KAN-16-32-8-B4096 | infer | gpu_native | 0.094 | 0.075 | 1.26x | 5.4 | 5.4 | cuda |
| KAN-16-32-8-B4096 | fwd_bwd | gpu_ref | 1.377 | 1.419 | 0.97x | 54.0 | 54.0 | reference |
| KAN-16-32-8-B4096 | fwd_bwd | gpu_ref_compile | 0.391 | 0.315 | 1.24x | 36.8 | 36.8 | reference |
| KAN-16-32-8-B4096 | fwd_bwd | gpu_native | 0.326 | 0.275 | 1.18x | 21.8 | 14.8 | cuda |
| KAN-16-32-8-B4096 | fwd_bwd_reg | gpu_ref | 4.125 | 3.879 | 1.06x | 92.4 | 111.9 | reference |
| KAN-16-32-8-B4096 | fwd_bwd_reg | gpu_ref_compile | 0.760 | 1.023 | 0.74x | 46.9 | 77.4 | reference |
| KAN-16-32-8-B4096 | fwd_bwd_reg | gpu_native | 2.172 | 1.081 | 2.01x | 57.0 | 18.4 | cuda |
| KAN-16-32-8-B4096 | fwd_bwd_reg_fused | gpu_ref | 4.125 | 2.499 | 1.65x | 92.4 | 69.7 | reference |
| KAN-16-32-8-B4096 | fwd_bwd_reg_fused | gpu_ref_compile | 0.760 | 0.900 | 0.84x | 46.9 | 46.9 | reference |
| KAN-16-32-8-B4096 | fwd_bwd_reg_fused | gpu_native | 2.172 | 0.763 | 2.85x | 57.0 | 15.9 | cuda |
| KAN-16-32-8-B4096 | grid_update | gpu_ref | 21.993 | 23.234 | 0.95x | 41.0 | 39.5 | reference |
| KAN-16-32-8-B4096 | grid_update | gpu_native | 21.268 | 21.826 | 0.97x | 41.0 | 39.5 | cuda |

CPU reference (wall ms, median):

| config | scenario | before | after |
|---|---|---:|---:|
| L-B1-16-G5 | infer | 0.113 | 0.120 |
| L-B1-16-G5 | fwd_bwd | 0.175 | 0.182 |
| L-B1-16-G5 | fwd_bwd_reg | 0.458 | 0.466 |
| L-B1-16-G5 | fwd_bwd_reg_fused | 0.458 | 0.334 |
| L-B256-64-G5 | infer | 0.684 | 0.710 |
| L-B256-64-G5 | fwd_bwd | 0.834 | 0.847 |
| L-B256-64-G5 | fwd_bwd_reg | 6.602 | 6.349 |
| L-B256-64-G5 | fwd_bwd_reg_fused | 6.602 | 6.041 |
| L-B256-64-G5 | grid_update | 6.154 | 6.707 |
| L-B1024-256-G5 | infer | 13.424 | 12.487 |
| L-B1024-256-G5 | fwd_bwd | 13.500 | 12.519 |
| L-B1024-256-G5 | fwd_bwd_reg | 566.388 | 545.068 |
| L-B1024-256-G5 | fwd_bwd_reg_fused | 566.388 | 530.233 |
| L-B1024-256-G5 | grid_update | 269.056 | 249.467 |
| L-B256-64-G64 | infer | 1.286 | 1.616 |
| L-B256-64-G64 | fwd_bwd | 2.093 | 1.933 |
| L-B256-64-G64 | fwd_bwd_reg | 10.090 | 9.225 |
| L-B256-64-G64 | fwd_bwd_reg_fused | 10.090 | 7.558 |
| L-B256-64-G64 | grid_update | 15.494 | 14.837 |
| KAN-16-32-8-B256 | infer | 0.773 | 0.869 |
| KAN-16-32-8-B256 | fwd_bwd | 1.902 | 1.727 |
| KAN-16-32-8-B256 | fwd_bwd_reg | 4.558 | 4.460 |
| KAN-16-32-8-B256 | fwd_bwd_reg_fused | 4.558 | 3.041 |
| KAN-16-32-8-B256 | grid_update | 4.679 | 4.580 |
| KAN-16-32-8-B4096 | infer | 4.992 | 3.992 |
| KAN-16-32-8-B4096 | fwd_bwd | 13.351 | 10.528 |
| KAN-16-32-8-B4096 | fwd_bwd_reg | 35.051 | 39.262 |
| KAN-16-32-8-B4096 | fwd_bwd_reg_fused | 35.051 | 26.634 |
| KAN-16-32-8-B4096 | grid_update | 34.553 | 33.801 |

Training-step memory breakdown (Adam, optimizer.step executed), native backend:

| config | scenario | side | params | grads | Adam state | peak inc | transient |
|---|---|---|---:|---:|---:|---:|---:|
| L-B1-16-G5 | fwd_bwd_reg | before | 0.01 | 0.01 | 0.02 | 0.0 | 0.0 |
| L-B1-16-G5 | fwd_bwd_reg_fused | after | 0.01 | 0.01 | 0.02 | 0.0 | 0.0 |
| L-B1-16-G5 | fwd_bwd_reg | after (separate API) | 0.01 | 0.01 | 0.02 | 0.0 | 0.0 |
| L-B256-64-G5 | fwd_bwd_reg | before | 0.16 | 0.16 | 0.31 | 17.3 | 17.2 |
| L-B256-64-G5 | fwd_bwd_reg_fused | after | 0.16 | 0.16 | 0.31 | 0.9 | 0.8 |
| L-B256-64-G5 | fwd_bwd_reg | after (separate API) | 0.16 | 0.16 | 0.31 | 1.4 | 1.3 |
| L-B1024-256-G5 | fwd_bwd_reg | before | 2.50 | 2.50 | 5.00 | 1141.3 | 1138.8 |
| L-B1024-256-G5 | fwd_bwd_reg_fused | after | 2.50 | 2.50 | 5.00 | 15.0 | 12.5 |
| L-B1024-256-G5 | fwd_bwd_reg | after (separate API) | 2.50 | 2.50 | 5.00 | 22.5 | 20.0 |
| L-B256-64-G64 | fwd_bwd_reg | before | 1.08 | 1.08 | 2.16 | 34.5 | 33.4 |
| L-B256-64-G64 | fwd_bwd_reg_fused | after | 1.08 | 1.08 | 2.16 | 6.4 | 5.4 |
| L-B256-64-G64 | fwd_bwd_reg | after (separate API) | 1.08 | 1.08 | 2.16 | 10.6 | 9.5 |
| L-B4096-256-G5 | fwd_bwd_reg | before | 2.50 | 2.50 | 5.00 | 4302.3 | 4299.8 |
| L-B4096-256-G5 | fwd_bwd_reg_fused | after | 2.50 | 2.50 | 5.00 | 51.0 | 48.5 |
| L-B4096-256-G5 | fwd_bwd_reg | after (separate API) | 2.50 | 2.50 | 5.00 | 79.3 | 76.8 |
| KAN-16-32-8-B256 | fwd_bwd_reg | before | 0.03 | 0.03 | 0.06 | 3.3 | 3.3 |
| KAN-16-32-8-B256 | fwd_bwd_reg_fused | after | 0.03 | 0.03 | 0.06 | 1.0 | 1.0 |
| KAN-16-32-8-B256 | fwd_bwd_reg | after (separate API) | 0.03 | 0.03 | 0.06 | 1.2 | 1.2 |
| KAN-16-32-8-B4096 | fwd_bwd_reg | before | 0.03 | 0.03 | 0.06 | 57.0 | 57.0 |
| KAN-16-32-8-B4096 | fwd_bwd_reg_fused | after | 0.03 | 0.03 | 0.06 | 15.9 | 15.9 |
| KAN-16-32-8-B4096 | fwd_bwd_reg | after (separate API) | 0.03 | 0.03 | 0.06 | 18.4 | 18.4 |
