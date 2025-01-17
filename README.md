# Speed Benchmark of Attention Implementation for LayoutLMv3

## Results

The following benchmark results were obtained using an NVIDIA A100 GPU with 80 GB of VRAM. The tests were conducted for training in `bfloat16` precision, using 16 workers, and the reported figures are based on the average of 2 runs, each consisting of several hundreds batches of random input.
The table below presents the throughput (in iterations per second), speed-up ratios, and VRAM consumption for each configuration.

| Batch Size | Seq Length | Eager (it/s) | SDPA (it/s) | Flash Attention v2 (it/s) | SDPA vs Eager | Flash v2 vs Eager | Eager VRAM (GB) | SDPA VRAM (GB) | Flash v2 VRAM (GB) |
|------------|------------|--------------|-------------|---------------------------|---------------|-------------------|-----------------|----------------|---------------------|
| 32         | 128        | 23.40        | 29.35       | <TBA>                     | 1.25x         | <TBA>             | 5.13            | 4.41           | <TBA>               |
| 32         | 512        | 4.36         | 10.35       | <TBA>                     | 2.37x         | <TBA>             | 21.61           | 8.85           | <TBA>               |
| 32         | 1024       | 1.41         | 5.63        | <TBA>                     | 3.99          | <TBA>             | 65.54           | 14.93          | <TBA>               |
| 64         | 128        | 13.92        | 19.80       | <TBA>                     | 1.42x         | <TBA>             | 7.21            | 5.87           | <TBA>               |
| 64         | 512        | 2.30         | 5.90        | <TBA>                     | 2.57x         | <TBA>             | 40.58           | 14.95          | <TBA>               |
| 64         | 1024       | *OOM*        | 2.98        | <TBA>                     | -x            | <TBA>             | *OOM*           | 26.91          | <TBA>               |
| 128        | 128        | 7.77         | 11.48       | <TBA>                     | 1.48x         | <TBA>             | 11.97           | 8.85           | <TBA>               |
| 128        | 512        | 1.18         | 3.21        | <TBA>                     | 2.72x         | <TBA>             | 80.09           | 26.91          | <TBA>               |
