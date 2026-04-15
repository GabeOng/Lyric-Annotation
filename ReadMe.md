1. adjust number of songs you want to train on in download.py, (UPDATED APR 2: skips songs if already exists, no need to delete audio folder)

2. run download.py, this will create and populate the audio folder 

3. delete current processed folder at DALI/DALI_v1.0 if it exists, then run processMfcc.py. This will convert the audio files we downloaded into mfcc's, which we use as the trainable inputs

4. run train.py, this will train the model. you can adjust the number of epochs, batch size, and learning rate at the bottom in the function call

5. run evaluate.py, this will evaluate the model by running it on all the songs and computing the wer for each, then compute the average wer at the end. We could potentially use bleu score to evaluate this in the future.

also, right now it only tests on the training data, we have no training/testing data split yet.

