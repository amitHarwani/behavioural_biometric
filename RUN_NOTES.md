200x46 | Supervised Contrastive Loss
Train sequences 358149 Val sequences 3305
# Run 1
steps_per_epoch 22387 total_steps_to_train 223870
NOTE: LR Bug max | 6e-4, min 6e-5, warmup: 10%

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
NOTE: LR Bug | max: 6e-4, min 6e-5, warmup: 10%

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

# Run 3 - Post Layer Norm and gradient clip of 1.0
NOTE: LR Bug | max: 6e-4, min 6e-5, warmup: 10%
Val EER: 
Epoch 0: 30.2621 | 27.7691
Epoch 1: 24.6340 | 22.1556
Epoch 2: 23.0703 | 20.0034
Epoch 3: 20.9465 | 16.9876
Epoch 4: 23.2974 | 16.5349
Epoch 5: 19.2823 | 13.6250
**Epoch 6: 18.0946 | 12.9079**
Epoch 7: 22.7152 | 15.0961
Epoch 8: 20.8079 | 16.0207
Epoch 9: 22.9683 | 16.1532
 -- Solved LR Bug After This

# Run 4 - max: 1e-4, min 1e-5 | 10 epochs (Warmup 10%)
Val EER: 
Epoch 0: 24.836667586191233 | 23.851515265366952
Epoch 1: 23.02210722116805  | 21.224587470362195
Epoch 2: 23.093345966569782 | 19.867675971864507
Epoch 3: 22.30624393988986  | 19.053686169551977
Epoch 4: 23.162076134687815 | 18.876382831523774
Epoch 5: 22.317708004909868 | 17.22006020022699
**Epoch 6: 21.164612833449915 | 17.02447747906897**
Epoch 7: 20.5719903359796   | 16.715986762590777
Epoch 8: 19.942691197502374 | 16.943626624170516
Epoch 9: 19.791206491307864 | 17.057708262843242

# Run 5 - Linear warmup to 1e-4 (20% - 2 epochs), and stay there
Val EER: 
Epoch 0: 31.1945 | 27.0056
Epoch 1: 21.8721 | 21.8855
Epoch 2: 23.1222 | 19.7236
Epoch 3: 24.6599 | 20.4397
Epoch 4: 24.6868 | 19.9725
Epoch 5: 24.7200 | 19.8537
Epoch 6: 23.4917 | 17.9986
Epoch 7: 21.0048 | 16.1999
Epoch 8: 22.0819 | 17.5746
Epoch 9: 21.4700 | 15.7000
Epoch 10: 19.8933 | 15.5562
Epoch 11: 20.5696 | 15.2178
Epoch 12: 19.1102 | 14.9506
Epoch 13: 19.8024 | 14.7911
Epoch 14: 18.2553 | 14.2376
Epoch 15: 19.1275 | 15.0761
Epoch 16: 19.5805 | 14.2986
Epoch 17: 19.1866 | 15.0476
Epoch 18: 19.1552 | 14.4487
**Epoch 19: 17.7967 | 13.3554**

# Run 6 - Linear warmup to 1e-3 (20% - epochs), and stay there - No gradient clipping: 
Epoch 0: 23.7832 | 20.6077
Epoch 1: 21.2346 | 16.9010
Epoch 2: 20.7503 | 15.5836
**Epoch 3: 21.2410 | 12.6068**
Epoch 4: 23.2601 | 14.4516
Epoch 5: 24.0379 | 18.6179
Epoch 6: 25.3768 | 17.6301
Epoch 7: 23.3631 | 18.8113


# Run - Temperature: 0.05 and maybe higher
# Run - 128 embedding 
# Run - Flattening instead of CLS Token
# Run - More Layers, More Heads
# Run - Channel Attention
# Run - CNN 

