
50 Keystrokes - A G M - 

45 + 5
45 + 5



10ms sequences - If keystroke is not present zeroed out

200 sequences combined to form a 2 second sequence.
200x46 
Supervised Contrastive Loss

Train sequences 358149 (79 users) Val sequences 3305 (10 users)

200,000 Parameters
# Run 1
steps_per_epoch 22387 total_steps_to_train 223870
NOTE: LR Bug max | 6e-4, min 6e-5, warmup: 10%
Each class in batch has 2 samples Batch 128
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

Observation - Poor separation , Better performance of Mahalonobis

# Run 2
Changed to Class Balanced Batch Sampler with K classes and C samples per class. 
4 classes 4 samples
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

Observation - Improvement because of the new sampler, Better Performance of Mahalonobis, Still Noisy

# Run 3 - Post Layer Norm and gradient clip of 1.0
NOTE: LR Bug | max: 6e-4, min 6e-5, warmup: 10%
4 classes 4 samples - (128 Actual Batch Size)
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

Observation - Post layer norm makes it more smooth, Found LR Bug


 -- Solved LR Bug After This

# Run 4 - max: 1e-4, min 1e-5 | 10 epochs (Warmup 10%)
4 classes 4 samples - (128 Actual Batch Size)
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

Observation: Set LR to 1e-4 to achieve same performance as run 3 but with the LR bug fices, However the eer does not improve after epoch 7
Maybe the learning rate drop is too much

# Run 5 - Linear warmup to 1e-4 (20% - 2 epochs), and stay there
4 classes 4 samples - (128 Actual Batch Size)
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

Observation: Setting learning rate to a constant of 1e-4 after warmup of 20%/2epochs, makes it smooth with slight bumps, But it takes
20 epochs to get to 13% eer

# Run 6 - Linear warmup to 1e-3 (20% - 2 epochs), and stay there - No gradient clipping: 
4 classes 4 samples - (128 Actual Batch Size)
Epoch 0: 23.7832 | 20.6077
Epoch 1: 21.2346 | 16.9010
Epoch 2: 20.7503 | 15.5836
**Epoch 3: 21.2410 | 12.6068**
Epoch 4: 23.2601 | 14.4516
Epoch 5: 24.0379 | 18.6179
Epoch 6: 25.3768 | 17.6301
Epoch 7: 23.3631 | 18.8113

Observation: Since the previous run took a long time, Setting the warmup to 1e-3 and then staying there with no gradient clipping - Helps achieve
the previous highest performance of 12%


-----------------------------------------------

## exp1
Cross Entropy Loss: train_1.py

- train_1.pt: 

    - Training on 10 user dataset (v1_merged_test_users_data) [Cross Entropy Loss] [BatchSize: 32]
        - Divided it also to train and test sequences with 20% test sequences.
    - Tested on 10 users (v1_merged_validation_users_data)   
        - 50 Squences Per Session: 50% for enrollment and 50% for verify: Mahaloobis EER: 10.02%
        - All Imp. Sequences: 50% for enrollment and 50% for verify: Mahalonobis EER: 7.74%

- train_1_1.pt
    - max_lr = 6e-4, min_lr = 6e-5, warm_up_steps = 2 epochs
    - Training on 79 user dataset (v1_merged_training_users_data) [Cross Entropy Loss] [BatchSize: 32]
        - Divided it also to train and test sequences with 20% test sequences.
    - Tested on 10 users (v1_merged_validation_users_data) 
        - During Training: 50 Seq per session: 50% for enrollment and 50% for verify: Mahalonobis EER: 7.73%
        - All Imp. Sequences: 50% for enrollment and 50% for verify:  **5.4212**
    - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **4.9671**
        - When taking equal number of imposter sequences per user: **3.8478**

    - Tried 2 Layers -> 5.57 X


## exp 2
Cross Entropy Loss: train_2.py

- train_2_0_epoch_9.pt: (6e-4 to 6e-5)
    -  Training on 79 user dataset (v1_merged_training_users_data) [Cross Entropy Loss] [BatchSize: 32]
        - Add dropout of 0.2, and weight decay of 0.1, and trained on the entire 79 users (No test sequence division)
     - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **3.0841**
        - When taking equal number of imposter sequences per user: **2.9946**

- train_2_0_epoch_19.pt:
    - Continued training from the above to 20 epochs, max_lr: 6e-5, min_lr: 6e-6(Continuation)
    - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **2.5005**
        - When taking equal number of imposter sequences per user: **2.3303**

- train_2_1_epoch_49.pt:
    - First 10 epochs: 6e-4 to 6e-5 (2 epoch warmup), Next 10 epochs: 6e-5 to 6e-6, Next 30 epochs: 6e-6 to 6e-7
    - All Imp. Sequences: **2.4751**
    - Equal Num.: **2.3797**

# exp 3
Cross Entropy Loss: train_3.py: Changed Dropout to 0.3.

6e-4 to 6e-5 (first 10 epoch) | 6e-5 to 6e-6 (next 10 epoch) | 6e-5 to 6e-6 (next 10 epochs)
- train_3_0_epoch_19.pt: 
    - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **2.5797**
        - When taking equal number of imposter sequences per user: **1.8637**

- train_3_0_epoch_29.pt:
    - 6e-5 to 6e-6 again
    - All Imp: 2.3928

# exp 4
Cross Entropy Loss: train_4.py:  Dropout 0.2, Add Channel Head Attention

6e-4 to 6e-5 (First 20 epoch) | 6e-5 to 6e-6 (Next 10 epoch)
- train_4_0_epoch_29.pt: 
    - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **2.4347**
        - When taking equal number of imposter sequences per user: **2.1603**


# exp5:  With CNN instead of MLP
6e-4 to 6e-5 (First 20 epoch) | 6e-5 to 6e-6 (Next 10 epoch)
    - 2.93 (All Imp.) | 2.76(Equal Imp.)


Results: 
|  Exp| Maha. EER (All Imp.)| Maha. EER (Equal Imp.)|
|---|---|---|
|  exp 1 - 10 epochs - Normal Transformer (CE Loss)(train_1_1.pt)                     |4.9671|3.8478|
|  exp 2 - 20 epochs - Add Dropout 0.2, Weight Decay: 0.1(train_2_0_epoch_19.pt)      |2.5005|2.3303| 
|  exp 2_1 - 50 epochs - Dropout 0.2, Weight Decay: 0.1(train_2_1_epoch_49.pt)        |2.4751|2.3797| 
|  exp 3 - 20 epochs - Increase Dropout to 0.3(train_3_0_epoch_19.pt)                 |2.5797|1.8637|
|  exp 3 - 30 epochs - (train_3_0_epoch_29.pt)                                        |**2.3928**|**1.6916**|
|  exp 4 - 30 epochs - Dropout 0.2, Add Channel Head Attn (train_4_0_epoch_29.pt)     |2.4347|2.1603|
|  exp 5 - 30 epochs - Dropout 0.2, Add CNN from exp4 (train_5_0_epoch_29.pt)         |2.93|2.76|



Notes -
 
When 10 epochs were done from 6e-4 to 6e-5, Validation eer at the end of 10 epoch was 5.07 for exp2, and 4.86 for exp3
But when 20 epoches from 6e-4 to 6e-5 were done, Validation eer at the end of 10 epoch was 7.65 for exp 4 and 6.63 for exp 5

-- 
First 10 epoch: 6e-4 to 6e-5, 
Then go from 6e-5 to 6e-6 (Properly) over 40 epochs.
-----------------------------
[Test Dataset Stats]
User: 0: Num. of sessions: 8
         Session: 0: Number of Sequences: 309
         Session: 1: Number of Sequences: 292
         Session: 2: Number of Sequences: 311
         Session: 3: Number of Sequences: 386
         Session: 4: Number of Sequences: 356
         Session: 5: Number of Sequences: 348
         Session: 6: Number of Sequences: 571
         Session: 7: Number of Sequences: 482
User: 1: Num. of sessions: 8
         Session: 0: Number of Sequences: 702
         Session: 1: Number of Sequences: 649
         Session: 2: Number of Sequences: 708
         Session: 3: Number of Sequences: 685
         Session: 4: Number of Sequences: 875
         Session: 5: Number of Sequences: 760
         Session: 6: Number of Sequences: 659
         Session: 7: Number of Sequences: 635
User: 2: Num. of sessions: 8
         Session: 0: Number of Sequences: 8
         Session: 1: Number of Sequences: 5
         Session: 2: Number of Sequences: 0
         Session: 3: Number of Sequences: 4
         Session: 4: Number of Sequences: 44
         Session: 5: Number of Sequences: 10
         Session: 6: Number of Sequences: 7
         Session: 7: Number of Sequences: 3
User: 3: Num. of sessions: 8
         Session: 0: Number of Sequences: 522
         Session: 1: Number of Sequences: 431
         Session: 2: Number of Sequences: 457
         Session: 3: Number of Sequences: 397
         Session: 4: Number of Sequences: 427
         Session: 5: Number of Sequences: 485
         Session: 6: Number of Sequences: 446
         Session: 7: Number of Sequences: 453
User: 4: Num. of sessions: 8
         Session: 0: Number of Sequences: 1244
         Session: 1: Number of Sequences: 948
         Session: 2: Number of Sequences: 829
         Session: 3: Number of Sequences: 720
         Session: 4: Number of Sequences: 768
         Session: 5: Number of Sequences: 809
         Session: 6: Number of Sequences: 821
         Session: 7: Number of Sequences: 693
User: 5: Num. of sessions: 8
         Session: 0: Number of Sequences: 547
         Session: 1: Number of Sequences: 465
         Session: 2: Number of Sequences: 523
         Session: 3: Number of Sequences: 422
         Session: 4: Number of Sequences: 400
         Session: 5: Number of Sequences: 572
         Session: 6: Number of Sequences: 489
         Session: 7: Number of Sequences: 429
User: 6: Num. of sessions: 8
         Session: 0: Number of Sequences: 545
         Session: 1: Number of Sequences: 122
         Session: 2: Number of Sequences: 318
         Session: 3: Number of Sequences: 336
         Session: 4: Number of Sequences: 281
         Session: 5: Number of Sequences: 358
         Session: 6: Number of Sequences: 404
         Session: 7: Number of Sequences: 404
User: 7: Num. of sessions: 8
         Session: 0: Number of Sequences: 419
         Session: 1: Number of Sequences: 459
         Session: 2: Number of Sequences: 398
         Session: 3: Number of Sequences: 326
         Session: 4: Number of Sequences: 334
         Session: 5: Number of Sequences: 376
         Session: 6: Number of Sequences: 463
         Session: 7: Number of Sequences: 358
User: 8: Num. of sessions: 8
         Session: 0: Number of Sequences: 480
         Session: 1: Number of Sequences: 502
         Session: 2: Number of Sequences: 589
         Session: 3: Number of Sequences: 385
         Session: 4: Number of Sequences: 552
         Session: 5: Number of Sequences: 292
         Session: 6: Number of Sequences: 472
         Session: 7: Number of Sequences: 520
User: 9: Num. of sessions: 8
         Session: 0: Number of Sequences: 627
         Session: 1: Number of Sequences: 431
         Session: 2: Number of Sequences: 462
         Session: 3: Number of Sequences: 538
         Session: 4: Number of Sequences: 453
         Session: 5: Number of Sequences: 378
         Session: 6: Number of Sequences: 498
         Session: 7: Number of Sequences: 448

