#!/bin/bash

## PSM

dataname=PSM

# models1=( LSTM_VAE OmniAnomaly)
# models2=( DAGMM USAD TF AnomalyTransformer )
models2=( TF )

# for model in ${models1[@]}
# do
#  python main.py \
#  --train --test \
#  --model $model \
#  --dataname $dataname \
#  --log_to_wandb
# done

# for model in ${models2[@]}
# do
#  python main.py \
#  --train \
#  --test \
#  --model $model \
#  --dataname $dataname \
#  --use_multi_gpu \
#  --log_to_wandb \
#  --devices 0,1
# done



## SWaT
#
dataname=SWaT
#
#for model in ${models1[@]}
#do
#  python main.py \
#  --train --test \
#  --model $model \
#  --dataname $dataname \
#  --log_to_wandb
#done
#
for model in ${models2[@]}
do
 python main.py \
 --train \
 --test \
 --model $model \
 --dataname $dataname \
 --use_multi_gpu \
 --log_to_wandb \
 --devices 0,1
done


### SMD

# dataname=SMD
# subtypes=( machine-1 machine-2 machine-3 )

# for model in ${models1[@]}
# do
#   for subtype in ${subtypes[@]}
#   do
#     for var in {1..11}
#     do
#       subdataname=$subtype-$var
#     if [ -f "../data/ServerMachineDataset/train/$subdataname.txt" ]; then
#       python main.py \
#       --train \
#       --test \
#       --model $model \
#       --dataname $dataname \
#       --subdataname $subdataname \
#       --log_to_wandb
#     fi
#     done
#   done
# done
#
#for model in ${models2[@]}
#do
#  for subtype in ${subtypes[@]}
#  do
#    for var in {1..11}
#    do
#      subdataname=$subtype-$var
#    if [ -f "../data/ServerMachineDataset/train/$subdataname.txt" ]; then
#      python main.py \
#      --train \
#      --test \
#      --model $model \
#      --dataname $dataname \
#      --subdataname $subdataname \
#      --log_to_wandb \
#      --use_multi_gpu \
#      --devices 0,1
#    fi
#    done
#  done
#done

## SMAP

# dataname=SMAP
# subtypes=( A B D E F G P R S T)
# A=( 1 2 3 4 5 6 7 8 9 )
# B=( 1 )
# D=( 1 2 3 4 5 6 7 8 9 11 12 13 )
# E=( 1 2 3 4 5 6 7 8 9 10 11 12 13 )
# F=( 1 2 3 )
# G=( 1 2 3 4 6 7 )
# P=( 1 2 3 4 7 )
# R=( 1 )
# S=( 1 )
# T=( 1 2 3 )

# for model in ${models1[@]}
# do
#   for subtype in ${subtypes[@]}
#   do
#     eval "arr=( \${$subtype[@]} )"
#     for var in ${arr[@]}
#     do
#       subdataname=$subtype-$var
# #      echo $subdataname
#     if [ -f "../data/SMAP_MSL/train/$subdataname.npy" ]; then
#       python main.py \
#       --train \
#       --test \
#       --model $model \
#       --dataname $dataname \
#       --subdataname $subdataname \
#       --log_to_wandb
#     fi
#     done
#   done
# done

# for model in ${models2[@]}
# do
#   for subtype in ${subtypes[@]}
#   do
#     eval "arr=( \${$subtype[@]} )"
#     for var in ${arr[@]}
#     do
#       subdataname=$subtype-$var
# #      echo $subdataname
#     if [ -f "../data/SMAP_MSL/train/$subdataname.npy" ]; then
#       python main.py \
#       --train \
#       --test \
#       --model $model \
#       --dataname $dataname \
#       --subdataname $subdataname \
#       --log_to_wandb \
#       --use_multi_gpu \
#       --devices 0,1
#     fi
#     done
#   done
# done


# ### MSL

# dataname=MSL
# subtypes=( C D F M P S T )
# C=( 1 2 )
# D=( 14 15 16 )
# F=( 4 5 7 8 )
# M=( 1 2 3 4 5 6 7 )
# P=( 10 11 14 15 )
# S=( 2 )
# T=( 4 5 8 9 12 13)

# for model in ${models1[@]}
# do
#   for subtype in ${subtypes[@]}
#   do
#     eval "arr=( \${$subtype[@]} )"
#     for var in ${arr[@]}
#     do
#       subdataname=$subtype-$var
#     if [ -f "../data/SMAP_MSL/train/$subdataname.npy" ]; then
#       python main.py \
#       --train \
#       --test \
#       --model $model \
#       --dataname $dataname \
#       --subdataname $subdataname \
#       --log_to_wandb
#     fi
#     done
#   done
# done

# for model in ${models2[@]}
# do
#   for subtype in ${subtypes[@]}
#   do
#     eval "arr=( \${$subtype[@]} )"
#     for var in ${arr[@]}
#     do
#       subdataname=$subtype-$var
#     if [ -f "../data/SMAP_MSL/train/$subdataname.npy" ]; then
#       python main.py \
#       --train \
#       --test \
#       --model $model \
#       --dataname $dataname \
#       --subdataname $subdataname \
#       --log_to_wandb \
#       --use_multi_gpu \
#       --devices 0,1
#     fi
#     done
#   done
# done