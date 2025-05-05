### RUN 1

Data: 30s + 5s overlap
k = 18
dropout = 0.1
n_epochs = 100 with EarlyStopping of delta=0.001 and patience=10
max_lr = 5e-3
min_lr = 5e-4 (max_lr * 0.1)
warmup_steps = 10% (0.1)
training dataloader = sampler2
batch size = 128
Notes: No channel_attention_mask, Linear projection layer to match dimensions.
steps_per_epoch 1317 total_steps_to_train 131700

Num. of parameters 2100106
steps_per_epoch 1317 total_steps_to_train 131700
num decayed parameter tensors: 77, with 2,093,347 parameters
num non-decayed parameter tensors: 69, with 6,759 parameters

