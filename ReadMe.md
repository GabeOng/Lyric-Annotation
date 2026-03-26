1. adjust number of songs you want to train on in download.py, delete current audio folder at DALI/DALI_v1.0 if it exists

2. run download.py, this will create and populate the audio folder 

3. delete current processed folder at DALI/DALI_v1.0 if it exists, then run processMfcc.py.

4. run train.py, this will train the model.

5. run evaluate.py, this will evaluate the model by running it on all the songs and computing the wer for each, then compute the average wer at the end.


