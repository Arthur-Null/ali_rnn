import random
import sys
import time
from datetime import timedelta

from tqdm import tqdm

sys.path.append('../../..')
import pickle as pkl
import tensorflow as tf
from src.feature_engineering.feature_API import *
from src.util.data import get_id_list_by_day, submit
from sklearn.metrics import log_loss

f = open('feature.pkl', 'rb')
train_feature, train_label = pkl.load(f)
# test_feature, test_label = pkl.load(f)
id_dict = {
    'train': get_id_list_by_day([1, 2, 3, 4, 5, 6]),
    'validation': get_id_list_by_day([7]),
    'test': get_id_list_by_day([8])
}
data = fetch_data(id_dict=id_dict, config_path='feature_config', missing_handler='mean', del_other=False)
feature_size = data.get_feature_size()
train_size = data.train_size
valid_size = data.valid_size
test_size = data.test_size


def get_batch_(data, label, batch_size, seqlen, shuffle=True):
    assert len(data) == len(label)
    total_size = len(data)
    print('total steps:', int(np.ceil(total_size * 1. / batch_size)))
    if shuffle:
        random.seed(42)
        all = list(zip(data, label))
        random.shuffle(all)
        data, label = list(zip(*all))
    for i in range(int(np.ceil(total_size * 1. / batch_size))):
        batch_data = []
        batch_label = []
        seq_len = []
        for j in range(batch_size * i, min(batch_size * (i + 1), total_size)):
            if len(data[j]) < seqlen:
                seq_len.append(len(data[j]))
                batch_data.append(data[j] + [[0.0] * feature_size for k in range(seqlen - len(data[j]))])
                batch_label.append(label[j] + [-1.0 for k in range(seqlen - len(data[j]))])
            else:
                seq_len.append(seqlen)
                batch_data.append(data[j][len(data[j]) - seqlen:])
                batch_label.append(label[j][len(data[j]) - seqlen:])
        yield np.array(batch_data), np.array(seq_len), np.array(batch_label)


def save_batch(data, label, batch_size, seqlen, name, shuffle=True):
    f = open(name, 'wb')
    assert len(data) == len(label)
    total_size = len(data)
    print('total steps:', int(np.ceil(total_size * 1. / batch_size)))
    if shuffle:
        random.seed(42)
        all = list(zip(data, label))
        random.shuffle(all)
        data, label = list(zip(*all))
    for i in tqdm(range(int(np.ceil(total_size * 1. / batch_size)) - 1)):
        batch_data = []
        batch_label = []
        seq_len = []
        for j in range(batch_size * i, min(batch_size * (i + 1), total_size)):
            if len(data[j]) < seqlen:
                seq_len.append(len(data[j]))
                batch_data.append(data[j] + [[0.0] * feature_size for k in range(seqlen - len(data[j]))])
                batch_label.append(label[j] + [-1.0 for k in range(seqlen - len(data[j]))])
            else:
                seq_len.append(seqlen)
                batch_data.append(data[j][len(data[j]) - seqlen:])
                batch_label.append(label[j][len(data[j]) - seqlen:])
        pkl.dump((np.array(batch_data), np.array(seq_len), np.array(batch_label)), f)


def save_sparse_batch(data, label, batch_size, seqlen, name, shuffle=True):
    f = open(name, 'wb')
    assert len(data) == len(label)
    total_size = len(data)
    print('total steps:', int(np.ceil(total_size * 1. / batch_size)))
    if shuffle:
        random.seed(42)
        all = list(zip(data, label))
        random.shuffle(all)
        data, label = list(zip(*all))
    for i in tqdm(range(int(np.ceil(total_size * 1. / batch_size)) - 1)):
        batch_label = []
        seq_len = []
        indice = []
        value = []
        shape = [batch_size, seqlen, feature_size]
        for j in range(batch_size * i, min(batch_size * (i + 1), total_size)):
            if len(data[j]) < seqlen:
                seq_len.append(len(data[j]))
                batch_label.append(label[j] + [-1.0 for k in range(seqlen - len(data[j]))])
                for k in range(len(data[j])):
                    for item in range(feature_size):
                        if data[j][k][item] != 0.:
                            indice.append([j % batch_size, k, item])
                            value.append(data[j][k][item])
            else:
                seq_len.append(seqlen)
                batch_label.append(label[j][len(data[j]) - seqlen:])
                for k in range(seqlen):
                    t = len(data[j]) - seqlen
                    for item in range(feature_size):
                        if data[j][t][item] != 0.:
                            indice.append([j % batch_size, k, item])
                            value.append(data[j][t][item])

        pkl.dump(((indice, value, shape), np.array(seq_len), np.array(batch_label)), f)


def get_batch(data, label, batch_size, name):
    f = open(name, 'rb')
    assert len(data) == len(label)
    total_size = len(data)
    print('total steps:', int(np.ceil(total_size * 1. / batch_size) - 1))
    for i in range(int(np.ceil(total_size * 1. / batch_size) - 1)):
        batch_data, seq_len, batch_label = pkl.load(f)
        yield np.array(batch_data), np.array(seq_len), np.array(batch_label)


# def get_batch(data, label, batch_size, seqlen, shuffle=True):
#     assert len(data) == len(label)
#     total_size = len(data)
#     print('total steps:', int(np.ceil(total_size * 1. / batch_size)))
#     if shuffle:
#         random.seed(42)
#         all = list(zip(data, label))
#         random.shuffle(all)
#         data, label = list(zip(*all))
#     for i in range(int(np.ceil(total_size * 1. / batch_size)) - 1):
#         batch_data = []
#         batch_label = []
#         seq_len = []
#         for j in range(batch_size * i, batch_size * (i + 1)):
#             batch_data.append(data[j])
#             batch_label.append(label[j])
#             seq_len.append(seqlen[j])
#         yield np.array(batch_data), np.array(seq_len), np.array(batch_label)


class Rnn(object):
    def __init__(self, config):
        self.graph = tf.Graph()

        self._path = '%s_%s_%s' % (config['learning_rate'], config['hidden_size'], config['batch_size'])
        self.config = config
        self._save_path, self._logs_path = None, None
        self.batches_step = 0
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _define_inputs(self):
        shape = [None]
        shape.extend([self.config['seq_max_len'], feature_size])
        label_shape = [None]
        label_shape.extend([self.config['seq_max_len']])
        self.input = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input'
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=label_shape,
            name='labels'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate'
        )
        self.seqlen = tf.placeholder(
            tf.int32,
            shape=[None],
            name='seqlen'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _build_graph(self):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config['hidden_size'])
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob)
        rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell, output_size=1, activation=tf.sigmoid)
        initial_state = rnn_cell.zero_state(self.config['batch_size'], tf.float32)

        outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.input, initial_state=initial_state,
                                           sequence_length=self.seqlen)
        batchsize = tf.shape(outputs)[0]
        index = tf.range(0, batchsize) * self.config['seq_max_len'] + (self.seqlen - 1)
        output = tf.gather(tf.reshape(outputs, [-1, 1]), index)
        self.prediction = output
        label = tf.gather(tf.reshape(self.labels, [-1, 1]), index)
        cost = tf.reduce_mean(tf.losses.log_loss(label, output))
        self.cross_entropy = cost
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        self.train_step = train_op

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def train_one_epoch(self):
        batches = get_batch(train_feature[:train_size], train_label[:train_size], self.config['batch_size'],
                            "train")
        train_losses = []
        valid_losses = []
        for batch in tqdm(batches):
            data, seqlen, label = batch
            feed_dict = {
                self.input: data,
                self.seqlen: seqlen,
                self.labels: label,
                self.learning_rate: self.config['learning_rate'],
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.train_step, self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, pred = result
            train_losses.append(loss)

        batches = get_batch(train_feature[train_size:], train_label[train_size:], self.config['batch_size'],
                            "valid")
        for batch in tqdm(batches):
            data, seqlen, label = batch
            feed_dict = {
                self.input: data,
                self.seqlen: seqlen,
                self.labels: label,
                self.learning_rate: self.config['learning_rate'],
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.cross_entropy, self.prediction]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss, pred = result
            valid_losses.append(loss)
        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        print("Loss = " + "{:.4f}".format(valid_loss))
        return train_loss, valid_loss

    def train_until_cov(self):
        total_start_time = time.time()
        epoch = 1
        losses = []
        while epoch < self.config['epoch']:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch()
            self.log(epoch, result, prefix='train')
            losses.append(result[1])

            time_per_epoch = time.time() - start_time
            if epoch > 3 and losses[-1] >= losses[-2] >= losses[-3]:
                break
            print('Time per epoch: %s' % (
                str(timedelta(seconds=time_per_epoch))
            ))
            epoch += 1

            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))
        return losses[-3], epoch

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    # def test(self, epoch):
    #     batches = get_batch(test_feature, test_label, self.config['batch_size'], "test")
    #     prediction = []
    #     for batch in tqdm(batches):
    #         data, seqlen, label = batch
    #         feed_dict = {
    #             self.input: data,
    #             self.seqlen: seqlen,
    #             self.labels: label,
    #             self.learning_rate: self.config['learning_rate'],
    #             self.is_training: True,
    #             self.keep_prob: 0.5
    #         }
    #         fetches = [self.cross_entropy, self.prediction]
    #         result = self.sess.run(fetches, feed_dict=feed_dict)
    #         loss, pred = result
    #         prediction += pred
    #     id_dict = {
    #         'train': get_id_list_by_day([1, 2, 3, 4, 5, 6]),
    #         'validation': get_id_list_by_day([7]),
    #         'test': get_id_list_by_day([8])
    #     }
    #     data = fetch_data(id_dict=id_dict, config_path='feature_config', missing_handler='mean', del_other=False)
    #     submit(pred=prediction, instance_id=data.get_instance_id('test'), file='rnn_submit')

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s_%s_%s" % (
            self.logs_path, str(self.config['learning_rate']), str(self.config['batch_size']),
            str(self.config['hidden_size'])),
                    'a')
        fout.write(s + '\n')

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)


class sparse_Rnn(object):
    def __init__(self, config):
        self.graph = tf.Graph()

        self._path = '%s_%s_%s' % (config['learning_rate'], config['hidden_size'], config['batch_size'])
        self.config = config
        self._save_path, self._logs_path = None, None
        self.batches_step = 0
        self.cross_entropy, self.train_step, self.prediction = None, None, None
        with self.graph.as_default():
            self._define_inputs()
            self._build_graph()
            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self._initialize_session()

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = '%s/checkpoint' % self._path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return self._save_path

    @property
    def logs_path(self):
        if self._logs_path is None:
            logs_path = '%s/logs' % self._path
            if not os.path.exists(logs_path):
                os.makedirs(logs_path)
            self._logs_path = logs_path
        return self._logs_path

    def _define_inputs(self):
        shape = [None]
        shape.extend([self.config['seq_max_len'], feature_size])
        label_shape = [None]
        label_shape.extend([self.config['seq_max_len']])
        self.input = tf.sparse_placeholder(
            tf.float32,
            shape=shape,
            name='input'
        )
        self.labels = tf.placeholder(
            tf.float32,
            shape=label_shape,
            name='labels'
        )
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate'
        )
        self.seqlen = tf.placeholder(
            tf.int32,
            shape=[None],
            name='seqlen'
        )
        self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    def _build_graph(self):
        input = tf.sparse_tensor_to_dense(self.input)
        input = tf.reshape(input, [self.config['batch_size'], self.config['seq_max_len'], feature_size])
        rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config['hidden_size'])
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=self.keep_prob,
                                                 output_keep_prob=self.keep_prob)
        rnn_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell, output_size=1, activation=tf.sigmoid)
        initial_state = rnn_cell.zero_state(self.config['batch_size'], tf.float32)

        outputs, state = tf.nn.dynamic_rnn(rnn_cell, input, initial_state=initial_state,
                                           sequence_length=self.seqlen)
        # batchsize = tf.shape(outputs)[0]
        # index = tf.range(0, batchsize) * self.config['seq_max_len'] + (self.seqlen - 1)
        # output = tf.gather(tf.reshape(outputs, [-1]), index)
        # self.prediction = output
        # label = self.labels
        # self.label = label
        # self.prediction = output\
        outputs = tf.reshape(outputs, [-1])
        labels = tf.reshape(self.labels, [-1])
        loss = tf.losses.log_loss(labels, outputs, loss_collection=None)
        mask = tf.reshape(tf.sequence_mask(self.seqlen, self.config['seq_max_len']), [-1])
        cost = tf.reduce_mean(tf.boolean_mask(loss, mask))
        self.cross_entropy = cost
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])
        gvs = optimizer.compute_gradients(cost)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)
        self.train_step = train_op

    def _initialize_session(self, set_logs=True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.sess.run(self.initializer)
        if set_logs:
            logswriter = tf.summary.FileWriter
            self.summary_writer = logswriter(self.logs_path, graph=self.graph)

    def train_one_epoch(self):
        batches = get_batch(train_feature[:train_size], train_label[:train_size], self.config['batch_size'],
                            "train_sparse")
        losses = []
        for batch in tqdm(batches):
            data, seqlen, label = batch
            feed_dict = {
                self.input: data,
                self.seqlen: seqlen,
                self.labels: label,
                self.learning_rate: self.config['learning_rate'],
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.train_step, self.cross_entropy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss = result
            losses.append(loss)
        train_loss = np.mean(losses)

        batches = get_batch(train_feature[train_size:], train_label[train_size:], self.config['batch_size'],
                            "valid_sparse")
        losses = []
        for batch in tqdm(batches):
            data, seqlen, label = batch
            feed_dict = {
                self.input: data,
                self.seqlen: seqlen,
                self.labels: label,
                self.learning_rate: self.config['learning_rate'],
                self.is_training: True,
                self.keep_prob: 0.5
            }
            fetches = [self.cross_entropy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            loss = result
            losses.append(loss)
        valid_loss = np.mean(losses)
        print("Train_Loss = " + "{:.4f}".format(train_loss))
        print("Valid_Loss = " + "{:.4f}".format(valid_loss))
        return train_loss, valid_loss

    def train_until_cov(self):
        total_start_time = time.time()
        epoch = 1
        losses = []
        while epoch < self.config['epoch']:
            print('-' * 30, 'Train epoch: %d' % epoch, '-' * 30)
            start_time = time.time()

            print("Training...")
            result = self.train_one_epoch()
            self.log(epoch, result, prefix='train')
            losses.append(result[1])

            time_per_epoch = time.time() - start_time
            if epoch > 3 and losses[-1] >= losses[-2] >= losses[-3]:
                break
            print('Time per epoch: %s' % (
                str(timedelta(seconds=time_per_epoch))
            ))
            epoch += 1

            self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))
        return losses[-3], epoch

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    # def test(self, epoch):
    #     batches = get_batch(test_feature, test_label, self.config['batch_size'], "test_sparse")
    #     prediction = []
    #     for batch in tqdm(batches):
    #         data, seqlen, label = batch
    #         feed_dict = {
    #             self.input: data,
    #             self.seqlen: seqlen,
    #             self.labels: label,
    #             self.learning_rate: self.config['learning_rate'],
    #             self.is_training: True,
    #             self.keep_prob: 0.5
    #         }
    #         fetches = [self.cross_entropy, self.prediction]
    #         result = self.sess.run(fetches, feed_dict=feed_dict)
    #         loss, pred = result
    #         prediction += pred.tolist()
    #     id_dict = {
    #         'train': get_id_list_by_day([1, 2, 3, 4, 5, 6]),
    #         'validation': get_id_list_by_day([7]),
    #         'test': get_id_list_by_day([8])
    #     }
    #     data = fetch_data(id_dict=id_dict, config_path='feature_config', missing_handler='mean', del_other=False)
    #     submit(pred=prediction, instance_id=data.get_instance_id('test'), file='rnn_submit')

    def log(self, epoch, result, prefix):
        s = prefix + '\t' + str(epoch)
        for i in result:
            s += ('\t' + str(i))
        fout = open("%s/%s_%s_%s" % (
            self.logs_path, str(self.config['learning_rate']), str(self.config['batch_size']),
            str(self.config['hidden_size'])),
                    'a')
        fout.write(s + '\n')

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception:
            raise IOError('Failed to load model from save path: %s' % self.save_path)
        print('Successfully load model from save path: %s' % self.save_path)


if __name__ == '__main__':
    config_list = []
    for learning_rate in [1e-2, 1e-3, 1e-4, 1e-5]:
        for batch_size in [1500]:
            for hidden_size in [256, 512, 128]:
                config = {'learning_rate': learning_rate, 'batch_size': batch_size, 'hidden_size': hidden_size,
                          'seq_max_len': 50, 'epoch': 500}
                config_list.append(config)
    save_sparse_batch(train_feature[:train_size], train_label[:train_size], 1500, 50, 'train_sparse')
    save_sparse_batch(train_feature[train_size:], train_label[train_size:], 1500, 50, 'valid_sparse')
    # save_sparse_batch(test_feature, test_label, 1500, 50, 'test_sparse')

    for config in config_list:
        best_loss = 100000
        model = sparse_Rnn(config)
        loss, epoch = model.train_until_cov()
        if loss < best_loss:
            best_config = config
            best_config['epoch'] = epoch
            print(best_config, best_loss)
