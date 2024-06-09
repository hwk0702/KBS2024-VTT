#!/bin/bash

models=(VTTSAT VTTPAT)

### SWaT
for model in ${models[@]}
do
  python main.py \
  --train \
  --test \
  --model $model \
  --dataname SWaT \
  --use_multi_gpu \
  --devices 0,1,2,3
done

# ### SMD
for model in ${models[@]}
do
 for var in 1 2 3 4 5 6 7 8
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname SMD \
   --subdataname machine-1-$var
 done
done

for model in ${models[@]}
do
 for var in 1 2 3 4 5 6 7 8 9
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname SMD \
   --subdataname machine-2-$var
 done
done

for model in ${models[@]}
do
 for var in 1 2 3 4 5 6 7 8 9 10 11
 do
   python main.py \
   --train \
   --test \
   --model $model \
   --dataname SMD \
   --subdataname machine-3-$var
 done
done

# ## SMAP
for model in ${models[@]}
do
  for var in 1 2 3 4 5 6 7 8 9
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname A-$var
  done
done

for model in ${models[@]}
do
  python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname B-1
done

for model in ${models[@]}
do
  for var in 1 2 3 4 5 6 7 8 9 11 12 13
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname D-$var
  done
done
#
for model in ${models[@]}
do
  for var in 1 2 3 4 5 6 7 8 9 10 11 12 13
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname E-$var
  done
done

for model in ${models[@]}
do
  for var in 1 2 3
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname F-$var
  done
done

for model in ${models[@]}
do
  for var in 1 2 3 4 6 7
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname G-$var
  done
done

for model in ${models[@]}
do
  for var in 1 2 3 4 7
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname P-$var
  done
done

for model in ${models[@]}
do
  python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname R-1
done

for model in ${models[@]}
do
  python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname S-1
done

for model in ${models[@]}
do
  for var in 1 2 3
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname SMAP \
    --subdataname T-$var
  done
done

# ## MSL
for model in ${models[@]}
do
  for var in 1 2
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname C-$var
  done
done

for model in ${models[@]}
do
  for var in 14 15 16
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname D-$var
  done
done

for model in ${models[@]}
do
  for var in 4 5 7 8
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname F-$var
  done
done

for model in ${models[@]}
do
  for var in 1 2 3 4 5 6 7
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname M-$var
  done
done

for model in ${models[@]}
do
  for var in 10 11 14 15
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname P-$var
  done
done

for model in ${models[@]}
do
  for var in 2
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname S-$var
  done
done

for model in ${models[@]}
do
  for var in 4 5 8 9 12 13
  do
    python main.py \
    --train \
    --test \
    --model $model \
    --dataname MSL \
    --subdataname T-$var
  done
done