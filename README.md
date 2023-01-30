# mbart-benchmark
 Benchmarking throughput of MBART

## Step 1: Find optimal batch size

Probably best to run on a single-node, single-GPU:

```bash
OMP_NUM_THREADS=8 deepspeed \
        benchmark.py \
        configs/auto_bs_config.json \
        --output_dir ${output_dir}
```

## Step 2: Run benchmarks

```bash
OMP_NUM_THREADS=8 deepspeed \
        --hostfile ${hostfile_name} \
        benchmark.py \
        configs/train_config.json \
        --output_dir ${output_dir} \
        --deepspeed configs/ds_config_zero2.json
```
