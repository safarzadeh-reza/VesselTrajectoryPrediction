## 1. Data preprocessing

### Training trajectory data loading class

TrajectoryLoader.py
The ship trajectory data loading class includes the trajectory data loading method for training and the getBatch method for training each model.

* row2array(row): row reading method
Parse a line of the read csv file into ship track point data, the data type is np.float32
The data structure for training is five-dimensional data: Δtime, Δlng, Δlat, sog, cog

* loadTrajectoryData(file_name)
Load the ship trajectory data in the csv file of file_name into the trajectory variable of the trajectory data class.
The ship trajectory data point structure is: Δtime, Δlng, Δlat, sog, cog.
Then perform standardization and normalization of dispersion, normalization: (x - min)/(max - min)
The data used for normalization is stored in self.max_train_data and self.min_train_data

* getBatchBP(batch_size, bp_step)
Randomly select a batch of trajectory and real point data for BP model training.
The trajectory length is bp_step. The extracted trajectory needs to be continuous.
Returns:
seq: shape of [batch_size, bp_step, 5].
next_point: shape of [batch_size, 5].

* getBatchLSTM(batch_size, seq_length)
Randomly select a batch of trajectory and real point data for LSTM model training.
The track length is seq_length. The extracted trajectory needs to be continuous.
Returns:
seq: shape of [batch_size, seq_length, 5].
next_point: shape of [batch_size, 5].

* getBatchSeq2Seq(batch_size, encoder_lenght, decoder_lenght)
Randomly select a batch of source trajectory and target sequence data for LSTM model training.
The trajectory lengths are encoder_length and decoder_length. The total trajectory needs to be continuous.
(The length of seq_decoder here is actually decoder_length+1, because the last bit of seq_encoder is used for input)
Returns:
seq_encoder: shape of [batch_size, encoder_length, 5].
seq_decoder: shape of [batch_size, decoder_length+1, 5].

### Test track data loading class

TestLoader.py
The ship trajectory data loading class includes the trajectory data loading method for testing and the getBatch method for testing each model.

* row2array(row): row reading method
Parse a line of the read csv file into ship track point data, the data type is np.float32
The test data includes absolute coordinates, and the structure is seven-dimensional data: Δtime, Δlng, Δlat, sog, cog, lng, lat.

* getTestBP(batch_size, bp_step):
Randomly select a batch of trajectories and real point data for BP model testing.
The trajectory length is bp_step. The extracted trajectory needs to be continuous.
Returns the test sequence and sequence absolute coordinates.
Returns:
x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
x_coordinates: coordinates of seq. [lng, lat]

* getTestLSTM(batch_size, seq_length):
Randomly select a batch of trajectory and real point data for LSTM model testing.
The track length is seq_length. The extracted trajectory needs to be continuous.
Returns the test sequence and sequence absolute coordinates.
Returns:
x_test: sequence for testing. [Δtime, Δlng, Δlat, sog, cog]
x_coordinates: coordinates of seq. [lng, lat]

* getTestSeq2Seq(batch_size, encoder_lenght, decoder_lenght):
Randomly select a batch of source trajectory and target sequence data for LSTM model testing.
The trajectory lengths are encoder_length and decoder_length. The total trajectory needs to be continuous.
Returns the encoding sequence, the absolute coordinates of the encoding sequence, the decoding sequence, and the absolute coordinates of the decoding sequence.
Returns:
seq_encoder_test: encoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
seq_encoder_coordinates: coordinates of encoder seq. [lng, lat].
seq_decoder_test: decoder sequence for testing. [Δtime, Δlng, Δlat, sog, cog].
seq_decoder_coordinates: coordinates of decoder seq. [lng, lat].

## 2, Network model

###LSTM

Model: "lstm"
______________________________________________________________
Layer (type) Output Shape Param #

lstm_1 (LSTM) multiple 81408

output_layer (Dense) multiple 645
______________________________________________________________
Total params: 82,053
Trainable params: 82,053
Non-trainable params: 0
______________________________________________________________

Hyperparameters: batch_size, lstm_step, n_lstm
lstm layer: units = n_lstm, dropout=0.1
output layer: units = 5

call:
The initial state is zero tensor
Directly input the lstm layer, then input the output layer, and return.

### Encoder

Model: "encoder"
______________________________________________________________
Layer (type) Output Shape Param #

lstm (LSTM) multiple 68608
______________________________________________________________

Total params: 68,608
Trainable params: 68,608
Non-trainable params: 0
______________________________________________________________

Hyperparameters: batch_size, n_lstm
lstm layer: units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output layer: units = 5

call:
The initial state is zero tensor
Directly input the lstm layer and return lstm_out and states.

### Decoder

Model: "decoder"
______________________________________________________________
Layer (type) Output Shape Param #

lstm_1 (LSTM) multiple 68608

output_layer (Dense) multiple 645
______________________________________________________________
Total params: 69,253
Trainable params: 69,253
Non-trainable params: 0
______________________________________________________________

Hyperparameters: batch_size, n_lstm
lstm layer: units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output layer: units = 5

call:
The initial state comes from the input
Directly input the lstm layer, return lstm_out and states, lstm_out passes the output layer
Returns output and status.

### Attention

* Attention layer whose function is 'dot'

Model: "attention"  n_lstm = 128, attention_func = 'dot'
_________________________________________________________________
Layer (type)                 Output Shape              Param #

Total params: 0
Trainable params: 0
Non-trainable params: 0
_________________________________________________________________

* Attention layer whose function is 'general'

Model: "attention"  n_lstm = 128, attention_func = 'general'
_________________________________________________________________
Layer (type)                 Output Shape              Param #

dense (Dense)                multiple                  16512

Total params: 16,512
Trainable params: 16,512
Non-trainable params: 0
_________________________________________________________________

* Attention layer whose function is 'concat'

Model: "attention"  n_lstm = 128, attention_func = 'concat'
_________________________________________________________________
Layer (type)                 Output Shape              Param #

dense (Dense)                multiple                  32896

dense_1 (Dense)              multiple                  129

Total params: 33,025
Trainable params: 33,025
Non-trainable params: 0
_________________________________________________________________

### DecoderAttention

Model: "decoder_attention"   n_lstm = 128, attention_func = 'general'
_________________________________________________________________
Layer (type)                 Output Shape              Param #

attention (Attention)        multiple                  16512

lstm_1 (LSTM)                multiple                  68608

wc_layer (Dense)             multiple                  32896

output_layer (Dense)         multiple                  645
_________________________________________________________________
Total params: 118,661
Trainable params: 118,661
Non-trainable params: 0
_________________________________________________________________

Hyperparameters: batch_size, n_lstm, attention_func
lstm layer: units = n_lstm, dropout=0.1, return_sequences=True, return_state=True
output layer: units = 5

call:
The initial state comes from the input
Attention layer input: lstmout and encoder_output
Directly input the lstm layer, return lstm_out and states, lstm_out passes the output layer
Returns output and status.

## 3. Network training

### BPtrain.py

### LSTMtrain.py

1. Set the model, training parameters, and optimization method:
n_lstm, seq_length, lstm_step
learning_rate, batch_size, num_batches
Optimizer generally chooses Adam

2. Create and initialize the model
Build the model: lstm neural_net.
Preform input shape by input
You can then summarize the network.

3. Set up tensorboard & checkpoint
Tensorboard is used to visualize the training process
Checkpoint saves the parameters after training for easy reloading during testing.

4. Define the training process
Step processing method: Process the track data of (batch_size, seq_length, 5) in the form of step-by-step input
  (batch_size, seq_length - lstm_step+1, lstm_step*5)
LSTM training process:
Loss is the MSE of the decoder output sequence and the real sequence (only Δlng, Δlat two-dimensional data are selected here)
Optimize codec parameters

5. Load the training set for training
Instantiate track data loading class
Use the StepProcess function to reconstruct the track data into one input per step
Perform training and record a checkpoint every display_step batch.

### Seq2Seqtrain.py

1. Set the model, training parameters, and optimization method:
n_lstm, encoder_length, decoder_length
learning_rate, batch_size, num_batches
Optimizer generally chooses Adam
Note that it is found here that more batch_num is needed to train the model, which is currently estimated to be about 10w.

2. Create and initialize the model
Build the model: encoder and decoder.
Preform input shape by input
You can then summarize the network.

3. Set up tensorboard & checkpoint
Tensorboard is used to visualize the training process
Checkpoint saves the parameters after training for easy reloading during testing.

4. Define the Seq2Seq training process
The decoder section requires a loop to implement timing prediction, and the loss is accumulated and finally averaged.
Loss is the RMSE of the decoder output sequence and the real sequence (only Δlng, Δlat two-dimensional data are selected here)
Real values are used for prediction during recursive prediction (the effectiveness of Scheduled Sampling needs to be verified)
Optimize codec parameters

5. Load the training set for training
Instantiate track data loading class
Perform training and record a checkpoint every display_step batch.

### AttentionSeq2Seqtrain.py

1. Set the model, training parameters, and optimization method:
n_lstm, encoder_length, decoder_length
learning_rate, batch_size, num_batches
Optimizer generally chooses Adam

2. Create and initialize the model
Build the model: encoder_a and decoder_a.
Preform input shape by input
You can then summarize the network.

3. Set up tensorboard & checkpoint
Tensorboard is used to visualize the training process
Checkpoint saves the parameters after training for easy reloading during testing.

4. Define the Attention-Seq2Seq training process
The decoder section requires a loop to implement timing prediction, and the loss is accumulated and finally averaged.
Loss is the MSE of the decoder output sequence and the real sequence (only Δlng, Δlat two-dimensional data are selected here)
Real values are used for prediction during recursive prediction (the effectiveness of Scheduled Sampling needs to be verified)
At the same time, as the prediction progresses, the historical track data is updated.
Optimize codec parameters

5. Load the training set for training
Instantiate track data loading class
Perform training and record a checkpoint every display_step batch.

## 4. Model testing

### TestPoints.py

Multi-point prediction test to test the actual prediction ability of the model

1. Create a model and reload parameters
Overloaded models: lstm, encoder, decoder, encoder_a, decoder_a

2. Define forecasting methods

     *TestSeq2Seq
     Perform Seq2Seq trajectory prediction and return the average RMSE loss.

     * TestSeq2SeqAttention
     Perform AttentionSeq2Seq trajectory prediction and return the average RMSE loss.

     *TestLSTM
     Perform LSTM trajectory prediction and use time series loops to achieve multi-point prediction.
     Returns the average RMSE loss.

3. Load test data and set test parameters
Load the test trajectory set and set the source sequence length and target sequence length for prediction.

4. Make predictions
Randomly obtain a continuous trajectory sequence of batch_size, including the source sequence for prediction and the target sequence for verification.
Used to output absolute coordinates.
LSTM, Seq2Seq, and AttentionSeq2Seq are used for prediction respectively, and loss is the average RMSE.

### TestVisual.py

Prediction point visualization test

1. Create a model and reload parameters
Overloaded models: lstm, encoder, decoder, encoder_a, decoder_a

2. Define forecasting methods

     *TestSeq2Seq
     Perform Seq2Seq trajectory prediction, use time series loops to predict, and save prediction points
     Return prediction points and average loss
     Returns:
     pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
     loss [tensor]: Root Mean Squre Error loss of prediction of points.

     * TestSeq2SeqAttention
     Perform AttentionSeq2Seq trajectory prediction, use time series loops to predict, and save prediction points
     Return prediction points and average loss
     Returns:
     pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
     loss [tensor]: Root Mean Squre Error loss of prediction of points.

     *TestLSTM
     Perform LSTM trajectory prediction, use time series loops to predict, and save prediction points
     Return prediction points and average loss
     Returns:
     pred [np.array(pred)]: The prediction of points. shape of [seq_length, 5].
     loss [tensor]: Root Mean Squre Error loss of prediction of points.

3. Load test data and set test parameters
Load the test trajectory set and set the source sequence length and target sequence length for prediction.
Randomly obtain a continuous trajectory sequence of batch_size, including the source sequence for prediction and the target sequence for verification.

4. Make predictions
Use LSTM, Seq2Seq, and AttentionSeq2Seq models to predict respectively. Loss is the average RMSE, and the results are printed.
The distribution of prediction results is stored in pred_lstm, pred_seq2seq, and pred_aseq2seq for visualization.

5. Absolute coordinate recovery and visualization
Distribution restores and visualizes source sequences, real target sequences, and predicted target training.
Source coordinates: load, normalize, convert to list, visualize with simplekml.
Real coordinates: load, normalize, convert to list, visualize with simplekml.
Predicted coordinates: load, normalize, convert to list, convert relative coordinates to absolute coordinates, visualize with simplekml.