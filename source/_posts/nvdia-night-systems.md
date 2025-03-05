---
title: NVDIA Night Systems
date: 2024-10-02 16:52:14
tags:
category:
math: true
---

# NVDIA Night Systems

- 主要用來分析 GPU 的 Performance
- 因為有直覺得 Timeline，可以看某個程式在哪個時間點再做什麼，所以也可以看 CPU 執行的程式
- Single thread/ Multi-thread (pthread/OpenMP)
  - `srun -n1 -cX nsys profile <nsys options> ../your_program <program args>`
- MPI
  `srun -nX ../wrapper.sh ../your_program <program args>`

- wrapper.sh

```sh
  #! /bin/bash

  mkdir -p nsys_reports

  # Output to ../nsys_reports/rank_$N.nsys-rep
  nsys profile \
  -o "../nsys_reports/rank_$PMI_RANK.nsys-rep" \
  --mpi-impl openmpi \
  --trace mpi,ucx,osrt \
  $@
```

讓每個 Process 輸出到不同名稱的檔案，裡面的 PMI_RANK 就會自動填入對應的 rank

--trace <events>: cuda, mpi, ucs, nvtx, ...
--start-later X: X 秒之後再開始 profile，因為像是 initialization 就不重要，且通常只要監測幾秒就好，不用全看
--duration Y: profile 跑幾秒
--mpi-impl: openmpi (for OpenMPI)/ mpich (for Intel MPI)

# NVTX

可以結合 Nsystem，插偵來更好的監測結果，不過會有一些 Profile Overhead
