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

- train_2_1_epoch_49.pt (Fresh Run From Start):
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
    - Continued Training From train_3_0_epoch_19.pt : 6e-5 to 6e-6 again
    - All Imp: **2.3928**
    - Equal Num.: **1.6916**

- train_3_1_epoch_49.pt:
    - Continued Training From train_3_0_epoch_29.pt : 6e-6 to 6e-7 for 20 epochs
    - All Imp: **2.2767**
    - Equal Num.: **1.8508**

- train_3_2 (train_3_2_epoch_39):
    - 5 Layers Than the 1 layer of 3_1
    - 6e-4 to 6e-5 with 2 epoch warmup (First 10 epochs), 6e-5 to 6e-6 (Next 10), 6e-6 to 6e-7 (Next 30)
    - All Imp: **1.9106** 
    - Performance drops in epoch 49

# exp 4
Cross Entropy Loss: train_4.py:  Dropout 0.2, Add Channel Head Attention

- train_4_0_epoch_29.pt: 
    - 6e-4 to 6e-5 (First 20 epoch) | 6e-5 to 6e-6 (Next 10 epoch)
    - Tested on 10 users (v1_merged_test_users_data)
        - All Imp. Sequences: 50% for enrollment and 50% for verify: **2.4347**
        - When taking equal number of imposter sequences per user: **2.1603**

- train_4_1_epoch_49.pt (Fresh Run From Start):
    - First 10 epochs: 6e-4 to 6e-5 (2 epoch warmup), Next 10 epochs: 6e-5 to 6e-6, Next 30 epochs: 6e-6 to 6e-7
    - All Imp. Sequences: **2.5568**
    - Equal Num.: **2.2364**

# exp5:  With CNN instead of MLP

- train_5_0_epoch_29.pt: 
    - 6e-4 to 6e-5 (First 20 epoch) | 6e-5 to 6e-6 (Next 10 epoch)
    - All Imp. : **2.93**
    - Equal Imp: **2.76**
        


Results: 
|  Exp| Maha. EER (All Imp.)| Maha. EER (Equal Imp.)|
|---|---|---|
|  exp 1 - 10 epochs - Normal Transformer (CE Loss)(train_1_1.pt)                     |4.9671|3.8478|
|  exp 2 - 20 epochs - Add Dropout 0.2, Weight Decay: 0.1(train_2_0_epoch_19.pt)      |2.5005|2.3303| 
|  exp 2_1 - 50 epochs - Dropout 0.2, Weight Decay: 0.1(train_2_1_epoch_49.pt)        |2.4751|2.3797| 
|  exp 3 - 20 epochs - Increase Dropout to 0.3(train_3_0_epoch_19.pt)                 |2.5797|1.8637|
|  exp 3 - 30 epochs - (train_3_0_epoch_29.pt)                                        |2.3928|1.6916|
|  exp 3_1 - 50 epochs - (train_3_1_epoch_49.pt)                                      |2.2767|1.8508|
|  exp 3_2 - 40 epochs - (train_3_2_epoch_39.pt)                                      |**1.9106**|
|  exp 3_3 - 50 epochs - (train_3_3_epoch_49.pt) One Second                           |**2.1272**|
|  exp 4 - 30 epochs - Dropout 0.2, Add Channel Head Attn (train_4_0_epoch_29.pt)     |2.4347|2.1603|
|  exp 4_1 - 50 epochs - Dropout 0.3,  (train_4_1_epoch_49.pt)                        |2.5568|2.2364|
|  exp 5 - 30 epochs - Dropout 0.2, Add CNN from exp4 (train_5_0_epoch_29.pt)         |2.93|2.76|


Any run with _1 or greater than it => Proper learning rate when resuming

Notes -
 
When 10 epochs were done from 6e-4 to 6e-5, Validation eer at the end of 10 epoch was 5.07 for exp2, and 4.86 for exp3
But when 20 epoches from 6e-4 to 6e-5 were done, Validation eer at the end of 10 epoch was 7.65 for exp 4 and 6.63 for exp 5

-- 
First 10 epoch: 6e-4 to 6e-5, 
Then go from 6e-5 to 6e-6 (Properly) over 40 epochs.
-----------------------------
[Test Dataset Stats]
| User | Total Number of Sequences |
| ---- | ------------------------- |
| 0    | 3,055                     |
| 1    | 5,673                     |
| 2    | 81                        |
| 3    | 3,618                     |
| 4    | 6,832                     |
| 5    | 3,847                     |
| 6    | 2,768                     |
| 7    | 3,133                     |
| 8    | 3,792                     |
| 9    | 3,835                     |
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

