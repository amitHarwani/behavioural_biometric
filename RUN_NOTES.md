# Run 1
200x46 | Supervised Contrastive Loss
Train sequences 358149 Val sequences 3305
steps_per_epoch 22387 total_steps_to_train 223870
Val EER: 
Epoch 0: 28.7541 | 27.7799
Epoch 1: 29.5205 | 24.2686
Epoch 2: 28.1303 | 22.7326
Epoch 3: 28.1430 | 23.8538
Epoch 4: 28.8775 | 25.7022
Epoch 5: 25.3935 | 25.4484
Epoch 6: 25.2692 | 24.3067
Epoch 7: 25.1000 | 24.5251
Epoch 8: 28.8910 | 24.3952
Epoch 9: 28.7267 | 21.3920


# Run 2
Changed to Class Balanced Batch Sampler with K classes and C samples per class.
Add Layer Norm before modality_proj
Change actual_batch_size to 128

Val EER: 
Epoch 0: 24.9912 | 22.0923
Epoch 1: 18.9946 | 17.1340
Epoch 2: 18.7027 | 16.8859
Epoch 3: 19.6939 | 14.4030
Epoch 4: 21.9922 | 15.3172
Epoch 5: 20.6269 | 14.9965
Epoch 6: 23.1522 | 16.9091
Epoch 7: 22.8936 | 16.6188
Epoch 8: 26.6163 | 18.7738
Epoch 9: 26.1382 | 20.0503
