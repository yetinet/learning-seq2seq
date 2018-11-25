#!/usr/bin/env python3

import random
from sklearn.model_selection import train_test_split


# special symbols
start_symbolstart_s  = '^'
end_symbol = '$'
padding_symbol = '#'


def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    """Generates pairs of equations and solutions to them.
       Each equation has a form of two integers with an operator in between.
       Each solution is an integer with the result of the operaion.
        allowed_operators: list of strings, allowed operators.
        dataset_size: an integer, number of equations to be generated.
        min_value: an integer, min value of each operand.
        max_value: an integer, max value of each operand.
        result: a list of tuples of strings (equation, solution).
    """
    sample = []
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    for _ in range(dataset_size):
        operator = random.choice(allowed_operators)
        first_value = random.randint(min_value,max_value)
        second_value = random.randint(min_value, max_value)
        if operator == "-":
            solution = str(first_value - second_value)
            sample.append( (str(first_value) + str(operator) + str(second_value),
                            solution))
        elif operator == "+":
            solution = str(first_value + second_value)
            sample.append((str(first_value) + str(operator) + str(second_value),
                           solution))

    return sample



allowed_operators = ['+', '-']
dataset_size = 100000
data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)

train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# make word embedding dictionary and embedding lookup
word2id = {symbol:i for i, symbol in enumerate('^$#+-1234567890')}
id2word = {i:symbol for symbol, i in word2id.items()}



def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.
      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.
      result: a tuple of (a list of ids, an actual length of sentence).
    """
    end_symbol = '$'
    padding_symbol = '#'

    sent_len = len(sentence)
    sent_ids = [word2id[word] for word in sentence]
    # Need to know whether to truncate, pad or leave alone
    if sent_len >= padded_len:
        sent_ids = sent_ids[:padded_len-1]
        sent_ids.append(word2id[end_symbol])
        non_padded_length = len(sent_ids)
    else:
        sent_ids.append(word2id[end_symbol])
        non_padded_length = len(sent_ids)
        for _ in range(padded_len - sent_len -1):
            sent_ids.append(word2id[padding_symbol])

    return sent_ids, non_padded_length


def ids_to_sentence(ids, id2word):
    """ Converts a sequence of ids to a sequence of symbols.
          ids: a list, indices for the padded sequence.
          id2word:  a dict, a mapping from ids to original symbols.
          result: a list of symbols.
    """

    return [id2word[i] for i in ids]


def batch_to_ids(sentences, word2id, max_len):
    """Prepares batches of indices.
       Sequences are padded to match the longest sequence in the batch,
       if it's longer than max_len, then max_len is used instead.
        sentences: a list of strings, original sequences.
        word2id: a dict, a mapping from original symbols to ids.
        max_len: an integer, max len of sequences allowed.
        result: a list of lists of ids, a list of actual lengths.
    """

    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len

def generate_batches(samples, batch_size):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y


import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, GRUCell
from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper, TrainingHelper


class Seq2SeqModel(object):
    pass

def declare_placeholders(self):
    """Specifies placeholders for the model."""

    # Placeholders for input and its actual lengths.
    self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int32, name='input_batch')
    self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')

    # Placeholders for groundtruth and its actual lengths.
    self.ground_truth =  tf.placeholder(shape=(None, None), dtype=tf.int32, name='ground_truth_batch')
    self.ground_truth_lengths =  tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_truth_batch_lengths')

    self.dropout_ph = tf.placeholder_with_default(tf.cast(dropout_keep_probability, tf.float32), shape=[])
    self.learning_rate_ph =  tf.placeholder_with_default(tf.cast(learning_rate, tf.float32), shape=[])

Seq2SeqModel.__declare_placeholders = classmethod(declare_placeholders)



def create_embeddings(self, vocab_size, embeddings_size):
    """Specifies embeddings layer and embeds an input batch."""

    random_initializer = tf.random_uniform((vocab_size, embeddings_size), -1.0, 1.0)
    self.embeddings =  tf.Variable(initial_value=random_initializer, dtype=tf.float32, name="embeddings")

    # Perform embeddings lookup for self.input_batch.
    self.input_batch_embedded =  tf.nn.embedding_lookup(self.embeddings, self.input_batch)

Seq2SeqModel.__create_embeddings = classmethod(create_embeddings)



def build_encoder(self, hidden_size):
    """Specifies encoder architecture and computes its output."""

    # Create GRUCell with dropout.
    encoder_cell =  GRUCell(num_units=hidden_size)
    cell_w_dropout = DropoutWrapper(cell=encoder_cell, input_keep_prob=self.dropout_ph, dtype=tf.float32)

    # Create RNN with the predefined cell.
    _, self.final_encoder_state =  tf.nn.dynamic_rnn(cell=cell_w_dropout,
                                                     inputs=self.input_batch_embedded,
                                                     sequence_length=self.input_batch_lengths,
                                                     dtype=tf.float32)


Seq2SeqModel.__build_encoder = classmethod(build_encoder)


def build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
    """Specifies decoder architecture and computes the output.
        Uses different helpers:
          - for train: feeding ground truth
          - for inference: feeding generated output
        As a result, self.train_outputs and self.infer_outputs are created.
        Each of them contains two fields:
          rnn_output (predicted logits)
          sample_id (predictions).
    """

    # Use start symbols as the decoder inputs at the first time step.
    batch_size = tf.shape(self.input_batch)[0]
    # created vector with length=input_batch size. going to fill it in w start tokens
    start_tokens = tf.fill([batch_size], start_symbol_id)
    # adding a start token to the beginning of each ground truth (Y) sentence
    ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)

    # Use the embedding layer defined before to lookup embedings for ground_truth_as_input.
    # Doing this because we want to use the same embeddings to encode ground truth that
    # we learned from training
    self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings, ground_truth_as_input)

    # Create TrainingHelper for the train stage.
    train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                     self.ground_truth_lengths)

    # Create GreedyEmbeddingHelper for the inference stage.
    # You should provide the embedding layer, start_tokens and index of the end symbol.
    infer_helper =  tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embeddings,
                                                             start_tokens=start_tokens,
                                                             end_token=end_symbol_id)

    def decode(helper, scope, reuse=None):
        """Creates decoder and return the results of the decoding with a given helper."""

        with tf.variable_scope(scope, reuse=reuse):
            # Create GRUCell with dropout. Do not forget to set the reuse flag properly.
            decoder_cell =  tf.contrib.rnn.GRUCell(num_units=hidden_size,
                                                   dtype=tf.float32,
                                                   reuse=reuse,
                                                   name="gru_cell")

            # Create a projection wrapper. Turns hidden states into logits
            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)

            # Create BasicDecoder, pass the defined cell, a helper, and initial state.
            # The initial state should be equal to the final state of the encoder!
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                      helper=helper,
                                                      initial_state=self.final_encoder_state)

            # The first returning argument of dynamic_decode contains two fields:
            #   rnn_output (predicted logits)
            #   sample_id (predictions)
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, maximum_iterations=max_iter,
                                                              output_time_major=False, impute_finished=True)

            return outputs

    self.train_outputs = decode(train_helper, 'decode')
    self.infer_outputs = decode(infer_helper, 'decode', reuse=True)

Seq2SeqModel.__build_decoder = classmethod(build_decoder)


def compute_loss(self):
    """Computes sequence loss (masked cross-entopy loss with logits)."""

    weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
    logits = self.train_outputs.rnn_output
    targets = self.ground_truth  ## NOTE: IF SOMETHING DOESNT WORK ITS PROB THIS

    self.loss   =  tf.contrib.seq2seq.sequence_loss(logits,
                                                    targets,
                                                    weights,
                                                    average_across_timesteps=True,
                                                    average_across_batch=True,
                                                    softmax_loss_function=None,
                                                    name="s2sloss")

Seq2SeqModel.__compute_loss = classmethod(compute_loss)



# Loss optimizer
def perform_optimization(self):
    """Specifies train_op that optimizes self.loss."""

    self.train_op =  tf.contrib.layers.optimize_loss(self.loss,
                                                     global_step = tf.train.get_global_step(),
                                                     learning_rate = self.learning_rate_ph,
                                                     optimizer = 'Adam',
                                                     clip_gradients = 1.0)

Seq2SeqModel.__perform_optimization = classmethod(perform_optimization)



def init_model(self, vocab_size, embeddings_size, hidden_size,
               max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):
    self.__declare_placeholders()
    self.__create_embeddings(vocab_size, embeddings_size)
    self.__build_encoder(hidden_size)
    self.__build_decoder(hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id)

    # Compute loss and back-propagate.
    self.__compute_loss()
    self.__perform_optimization()

    # Get predictions for evaluation.
    self.train_predictions = self.train_outputs.sample_id
    self.infer_predictions = self.infer_outputs.sample_id

Seq2SeqModel.__init__ = classmethod(init_model)



# Train on batch
def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, learning_rate, dropout_keep_probability):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability
        }
    pred, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
    return pred, loss

Seq2SeqModel.train_on_batch = classmethod(train_on_batch)



# Two predict function, one with losses for training, and one without for just predicting
def predict_for_batch(self, session, X, X_seq_len):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len
        }
    pred = session.run([
            self.infer_predictions
        ], feed_dict=feed_dict)[0]
    return pred


def predict_for_batch_with_loss(self, session, X, X_seq_len, Y, Y_seq_len):
    feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len
        }
    pred, loss = session.run([
            self.infer_predictions,
            self.loss,
        ], feed_dict=feed_dict)
    return pred, loss

Seq2SeqModel.predict_for_batch = classmethod(predict_for_batch)
Seq2SeqModel.predict_for_batch_with_loss = classmethod(predict_for_batch_with_loss)

#################################################################################################
# Start Training

tf.reset_default_graph()

# initialize model
embeddings_size = 20
hidden_size = 512
max_iter = 7
start_symbol_id = word2id[start_symbolstart_s]
end_symbol_id = word2id[end_symbol]
padding_symbol_id = word2id[padding_symbol]
vocab_size = len(word2id.keys())

batch_size = 32
n_epochs = 1
learning_rate = .001
dropout_keep_probability = .5
max_len = 20

model = Seq2SeqModel(vocab_size=vocab_size,
                     embeddings_size=embeddings_size,
                     hidden_size=hidden_size,
                     max_iter=max_iter,
                     start_symbol_id=start_symbol_id,
                     end_symbol_id=end_symbol_id,
                     padding_symbol_id=padding_symbol_id)


n_step = int(len(train_set) / batch_size)

# Start Session
session = tf.Session()

# Always do this
session.run(tf.global_variables_initializer())

invalid_number_prediction_counts = []
all_model_predictions = []
all_ground_truth = []

print('Start training... \n')
for epoch in range(n_epochs):
    random.shuffle(train_set)
    random.shuffle(test_set)

    print('Train: epoch', epoch + 1)
    for n_iter, (X_batch, Y_batch) in enumerate(generate_batches(train_set, batch_size=batch_size)):
        ######################################
        ######### YOUR CODE HERE #############
        ######################################
        # prepare the data (X_batch and Y_batch) for training
        # using function batch_to_ids
        X_ids, X_sent_len = batch_to_ids(X_batch, word2id, max_len=max_len)
        Y_ids, Y_sent_len = batch_to_ids(Y_batch, word2id, max_len=max_len)

        # predictions
        predictions, loss =model.train_on_batch(session=session,
                                            X=X_ids,
                                            X_seq_len=X_sent_len,
                                            Y=Y_ids,
                                            Y_seq_len=Y_sent_len,
                                            learning_rate=learning_rate,
                                            dropout_keep_probability=dropout_keep_probability)

        if n_iter % 200 == 0:
            print("Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1, n_step, loss))

    X_sent, Y_sent = next(generate_batches(test_set, batch_size=batch_size))
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    # prepare test data (X_sent and Y_sent) for predicting
    # quality and computing value of the loss function
    # using function batch_to_ids
    X_sent, X_sent_len = batch_to_ids(X_sent, word2id, max_len=20)
    Y_sent, Y_sent_len = batch_to_ids(Y_sent, word2id, max_len=20)

    predictions, loss =  model.predict_for_batch_with_loss(session,
                                                     X_sent,
                                                     X_sent_len,
                                                     Y_sent,
                                                     Y_sent_len)

    print('Test: epoch', epoch + 1, 'loss:', loss, )
    for x, y, p in list(zip(X_sent, Y_sent, predictions))[:3]:
        print('X:', ''.join(ids_to_sentence(x, id2word)))
        print('Y:', ''.join(ids_to_sentence(y, id2word)))
        print('O:', ''.join(ids_to_sentence(p, id2word)))
        print('')

    model_predictions = []
    ground_truth = []
    invalid_number_prediction_count = 0
    # For the whole test set calculate ground-truth values (as integer numbers)
    # and prediction values (also as integers) to calculate metrics.
    # If generated by model number is not correct (e.g. '1-1'),
    # increase invalid_number_prediction_count and don't append this and corresponding
    # ground-truth value to the arrays.
    for X_batch, Y_batch in generate_batches(test_set, batch_size=batch_size):
        X, X_seq_len = batch_to_ids(X_batch, word2id, max_len)
        Y, Y_seq_len = batch_to_ids(Y_batch, word2id, max_len)

        preds = model.predict_for_batch(session,
                                  X,
                                  X_seq_len)

        for y, p in zip(Y, predictions):
            y_sent = ''.join(ids_to_sentence(y, id2word))
            y_sent = y_sent[:y_sent.find('$')]
            p_sent = ''.join(ids_to_sentence(p, id2word))
            p_sent = p_sent[:p_sent.find('$')]
            if p_sent.isdigit() or (p_sent.startswith('-') and p_sent[1:].isdigit()):
                model_predictions.append(int(p_sent))
                ground_truth.append(int(y_sent))
            else:
                invalid_number_prediction_count += 1

    all_model_predictions.append(model_predictions)
    all_ground_truth.append(ground_truth)
    invalid_number_prediction_counts.append(invalid_number_prediction_count)

print('\n...training finished.')


## Evaluation
from sklearn.metrics import mean_absolute_error

for i, (gts, predictions, invalid_number_prediction_count) in enumerate(zip(all_ground_truth,
                                                                            all_model_predictions,
                                                                            invalid_number_prediction_counts), 1):
    mae = mean_absolute_error(gts, predictions)
    print("Epoch: %i, MAE: %f, Invalid numbers: %i" % (i, mae, invalid_number_prediction_count))




