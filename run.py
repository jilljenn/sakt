from prepare import fraction, assistments
from tqdm import tqdm

import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict

import sys

params = {
  'dataset': 'assistments09',
  'num_layers': 1,
  'embed_size': 150,
  # 'hid_size': 100,
  'num_heads': 5,
  'dropout': 0.2,
  'batch_size': 128,
  # 'lr': 1e-3,
  'num_epochs': 10,
  # 'n_traces': 1000,
}
Args = namedtuple('Args', sorted(params))
args = Args(**params)

if args.dataset == 'fraction':
  actions, lengths, exercises, targets = fraction()
else:
  actions, lengths, exercises, targets = assistments()

# sys.exit(0)
  
nb_samples = len(actions)
indices = np.random.permutation(nb_samples)
i_train = indices[:round(0.8 * nb_samples)]
i_test = indices[round(0.8 * nb_samples):]

BUFFER_SIZE = 20000
BATCH_SIZE = args.batch_size

MAX_LENGTH = 100

# def filter_max_length(x, y, max_length=MAX_LENGTH):
#   return tf.logical_and(tf.size(x) <= max_length,
#                         tf.size(y) <= max_length)

# def tf_encode(pt, en):
#   return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

train_dataset = tf.data.Dataset.from_tensor_slices((actions[i_train], exercises[i_train], targets[i_train]))
test_dataset = tf.data.Dataset.from_tensor_slices((actions[i_test], exercises[i_test], targets[i_test]))

# # train_dataset = train_examples.map(tf_encode)
# train_dataset = train_dataset.filter(filter_max_length)
# # cache the dataset to memory to get a speedup while reading from it.
# train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.batch(len(i_test))

# val_dataset = val_examples.map(tf_encode)
# val_dataset = val_dataset.filter(filter_max_length).padded_batch(
#     BATCH_SIZE, padded_shapes=([-1], [-1]))

# This is why I hate tf.data
for _ in enumerate(train_dataset):
  dt = time.time()
  _
  print('Well', time.time() - dt)

nb_samples = len(actions)
indices = np.random.permutation(nb_samples)
for i in range(nb_samples // BATCH_SIZE):
  dt = time.time()
  i_batch = indices[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
  actions[i_batch]
  exercises[i_batch]
  targets[i_batch]
  print('Well', time.time() - dt)

act, exe, tar = next(iter(train_dataset))
print('Batch', act.shape, exe.shape, tar.shape)
# sys.exit(0)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

# pos_encoding = positional_encoding(50, 512)
# print (pos_encoding.shape)

# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

def create_padding_mask(seq):
  '''
  0 means no element in the sequence
  '''
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
# print('wonder', create_padding_mask(x))
# sys.exit(0)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

# x = tf.random.uniform((1, 3))
# temp = create_look_ahead_mask(x.shape[1])
# print(temp)
# sys.exit(0)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

np.set_printoptions(suppress=True)

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, q, training, mask):
    '''
    x = values = keys
    q = queries
    '''

    attn_output, _ = self.mha(x, x, q, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2

sample_encoder_layer = EncoderLayer(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)),
    tf.random.uniform((64, 43, 512)), False, None)

print('Shape output', sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    print('VOCAB SIZE', input_vocab_size)
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    print('pos encoding', self.pos_encoding.shape)

    self.exe_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, q, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    print('x shape', x.shape)
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    q = self.exe_embedding(q)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, q, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, 
                         dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, temp_input, training=False, mask=None)

print ('Encoder output', sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, pe_input, rate)

    # self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
    #                        target_vocab_size, pe_target, rate)

    self.final_layer = tf.keras.layers.Dense(1)
    
  def call(self, inp, exe, training, look_ahead_mask=None):

    enc_output = self.encoder(inp, exe, training, look_ahead_mask)  # (batch_size, inp_seq_len, d_model)
    # print('ENCODER', enc_output.shape)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    # dec_output, attention_weights = self.decoder(
    #     tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = tf.squeeze(self.final_layer(enc_output))  # (batch_size, tar_seq_len, target_vocab_size)
    # print('Final shape', final_output.shape)
    # sys.exit(0)
    
    return final_output#, attention_weights

sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048, 
    input_vocab_size=8500, target_vocab_size=8000, 
    pe_input=10000, pe_target=6000)

temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)

print('Input', temp_input.shape)
print('Output', temp_target.shape)
fn_out = sample_transformer(temp_input, temp_target, training=False, 
                               look_ahead_mask=None)

print('Final final shape', fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

# Add same values as SAKT paper

num_layers = args.num_layers
d_model = args.embed_size
dff = d_model  # 512
num_heads = args.num_heads

VOCAB = 1 + np.max(actions)
print('vocab', VOCAB)
input_vocab_size = VOCAB
target_vocab_size = MAX_LENGTH
dropout_rate = args.dropout

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.show()

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True, reduction='none')
loss_object = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  # mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  # mask = tf.cast(mask, dtype=loss_.dtype)
  # loss_ *= mask
  # print('wow', loss_.shape)

  # return loss_
  return tf.reduce_mean(loss_)

metrics = {}
for category in {'train', 'test'}:
  metrics[category + '_loss'] = tf.keras.metrics.Mean(name=category + '_loss')
  metrics[category + '_accuracy'] = tf.keras.metrics.BinaryAccuracy(name=category + '_accuracy')
  metrics[category + '_auc'] = tf.keras.metrics.AUC(name=category + '_auc')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=MAX_LENGTH, 
                          pe_target=MAX_LENGTH,
                          rate=dropout_rate)

def create_masks(tar):
  # Encoder padding mask
  # enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  # dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return combined_mask

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = args.num_epochs

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32)
]

@tf.function(input_signature=train_step_signature)  # 3 times faster (on Fraction dataset)
def train_step(inp, exe, tar):
  combined_mask = create_masks(inp)
  
  with tf.GradientTape() as tape:
    predictions = transformer(inp, exe, True, combined_mask)
    loss = loss_function(tar, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  proba = tf.sigmoid(predictions)

  metrics['train_loss'](loss)
  metrics['train_accuracy'](tar, proba)
  metrics['train_auc'](tar, proba)

@tf.function(input_signature=train_step_signature)  # 3 times faster (on Fraction dataset)  
def forward_step(inp, exe, tar):
  combined_mask = create_masks(inp)
  
  predictions = transformer(inp, exe, False, combined_mask)
  loss = loss_function(tar, predictions)  
  proba = tf.sigmoid(predictions)
  
  metrics['test_loss'](loss)
  metrics['test_accuracy'](tar, proba)
  metrics['test_auc'](tar, proba)

def all_metrics(metrics):
  return ' '.join('{}={:.4f}'.format(name, metric.result()) for name, metric in metrics.items())

# Start training

records = defaultdict(list)
for epoch in tqdm(range(EPOCHS)):
  start = time.time()

  for name in metrics:
    metrics[name].reset_states()
  
  for (batch, (inp, exe, tar)) in enumerate(train_dataset):
    # print('Batch', inp.shape, exe.shape, tar.shape)
    train_step(inp, exe, tar)
    
    if batch % 50 == 0:
      print('Avance')
    #  print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
    #      epoch + 1, batch, train_loss.result(), train_accuracy.result()))

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
  for inp, exe, tar in test_dataset:
    forward_step(inp, exe, tar)
    
  print ('Epoch {} {}'.format(epoch + 1, all_metrics(metrics)))

  for name in metrics:
    records[name].append(metrics[name].result())

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

for name, values in records.items():
  plt.plot(np.arange(len(records['train_loss'])), values, label=name)
plt.legend()
plt.show()
