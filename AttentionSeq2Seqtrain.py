import tensorflow as tf
import numpy as np
import random
import csv
import Model
from TrajectoryLoader import TrajectoryLoader

# parameters for traning
learnig_rate = 0.001
num_batches = 3000
batch_size = 128
display_step = 50
# parameters for seq2seq model
n_lstm = 128
encoder_length = 120
decoder_length = 60

attention_func1 = 'dot' 
attention_func2 = 'general' 
attention_func3 = 'concat'

# Choose Adam optimizer.
optimizer = tf.keras.optimizers.Adam(learnig_rate)

# Create and build encoder and decoder.
encoder_a = Model.Encoder(n_lstm, batch_size)
decoder_a = Model.DecoderAttention(n_lstm, batch_size, attention_func2)

x = np.zeros((batch_size, 1, 5), dtype=np.float32)
output = encoder_a(x)
decoder_a(x, output[1:], output[0])
encoder_a.summary()
decoder_a.summary()
decoder_a.attention.summary()

# restore the last checkpoint
checkpoint4 = tf.train.Checkpoint(EncoderAttention = encoder_a)
checkpoint4.restore(tf.train.latest_checkpoint('./SaveEncoderAttention'))

checkpoint5 = tf.train.Checkpoint(DecoderAttention = decoder_a)
checkpoint5.restore(tf.train.latest_checkpoint('./SaveDecoderAttention'))



# tensorboard
profiler_outdir = '.\\tensorboard'
# tf.summary.trace_on(profiler=True, profiler_outdir=profiler_outdir)
summary_writer = tf.summary.create_file_writer('tensorboard')
tf.summary.trace_on(profiler=True)

# checkpoint
checkpoint1 = tf.train.Checkpoint(EncoderAttention = encoder_a)
manager1 = tf.train.CheckpointManager(checkpoint1, directory = './SaveEncoderAttention', checkpoint_name = 'EncoderAttention.ckpt', max_to_keep = 10)
checkpoint2 = tf.train.Checkpoint(DecoderAttention = decoder_a)
manager2 = tf.train.CheckpointManager(checkpoint2, directory = './SaveDecoderAttention', checkpoint_name = 'DecoderAttention.ckpt', max_to_keep = 10)

def RunOptimization(source_seq, target_seq_in, target_seq_out, step):
    loss = 0
    decoder_length = target_seq_out.shape[1]
    with tf.GradientTape() as tape:
        encoder_outputs = encoder_a(source_seq)
        states = encoder_outputs[1:]
        history = encoder_outputs[0]
        y_sample = 0
        for t in range(decoder_length):
            # TODO scheduled sampling
            if t == 0 or random.randint(0,1) == 2:
                decoder_in = tf.expand_dims(target_seq_in[:, t], 1)
            else:
                decoder_in = tf.expand_dims(y_sample, 1)        
            logit, lstm_out, de_state_h, de_state_c, _= decoder_a(decoder_in, states, history)
            y_sample = logit
            history_new = tf.expand_dims(lstm_out, 1)
            history = tf.concat([history[:, 1:], history_new], 1)
            states = de_state_h, de_state_c
            # loss function : RSME 
            loss_0 = tf.keras.losses.MSE(target_seq_out[:, t, 1:3], logit[:, 1:3])
            loss += tf.sqrt(loss_0)
       
    variables = encoder_a.trainable_variables + decoder_a.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))  
    
    loss = tf.reduce_mean(loss) 
    loss = loss / decoder_length
    with summary_writer.as_default():
        tf.summary.scalar("loss", loss.numpy(), step = step)   
    return loss

# Load trajectory data.
seq2seq_loader = TrajectoryLoader()
seq2seq_loader.loadTrajectoryData("./DataSet/DataSet/Trajectory10w.csv")

for batch_index in range(1, num_batches+1):
    seq_encoder, seq_decoder = seq2seq_loader.getBatchSeq2Seq(batch_size, encoder_length, decoder_length)
    seq_decoder_in = seq_decoder[:, :decoder_length, :]
    seq_decoder_out = seq_decoder[:, 1:decoder_length+1, :]
    loss = RunOptimization(seq_encoder, seq_decoder_in, seq_decoder_out, batch_index)

    if batch_index % display_step == 0:
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        path1 = manager1.save(checkpoint_number = batch_index)
        path2 = manager2.save(checkpoint_number = batch_index)

with summary_writer.as_default():
    tf.summary.trace_export(name = "model_trace", step = 0, profiler_outdir = 'tensorboard')

