'''
Created on 2018-3-30

@author: Dezhi Wang
'''
import numpy as np
import random
from sklearn.utils.metaestimators import if_delegate_has_method

class BalanceDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        n_each = batch_size // n_labs   
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            for i1 in xrange(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_each, len_list[i1])]
                batch_x.append(x[batch_idx])
                batch_y.append(y[batch_idx])
                pointer_list[i1] += n_each
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

class RatioDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._verbose_ = verbose
            
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in xrange(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            if n_samples < 1000:
                lb_list += [idx]
            elif n_samples < 2000:
                lb_list += [idx] * 2
            elif n_samples < 3000:
                lb_list += [idx] * 3
            elif n_samples < 4000:
                lb_list += [idx] * 4
            else:
                lb_list += [idx] * 5
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        if self._verbose_ == 1:
            print("n_samples_list: %s" % (n_samples_list,))
            print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in xrange(n_labs)]
            
            for i1 in xrange(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1]
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

class RatioDataGeneratorMixup(object):
    def __init__(self, batch_size, type, te_max_iter=100, verbose=1, alpha=0.4, mixup=True, datagen=None):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._verbose_ = verbose
        self.alpha = alpha
        self.mixup = mixup
        self.datagen = datagen  
            
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in xrange(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            if n_samples < 800:
                lb_list += [idx]
            elif n_samples < 1500:
                lb_list += [idx] * 2
            elif n_samples < 2000:
                lb_list += [idx] * 3
            elif n_samples < 3000:
                lb_list += [idx] * 4
            else:
                lb_list += [idx] * 5
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        if self._verbose_ == 1:
            print("n_samples_list: %s" % (n_samples_list,))
            print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        index_list = []
        for i1 in xrange(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in xrange(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in xrange(n_labs)]
            
            for i1 in xrange(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1] ### pay attention
            
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)   
                
            if  self.mixup:
                weights = np.random.beta(self.alpha, self.alpha, batch_x.shape[0])
                
                indxran = np.random.permutation(batch_x.shape[0])
                x1, x2 = batch_x, batch_x[indxran]
            
                batch_x_min = np.zeros(x1.shape)
                for i in range(batch_x.shape[0]):
                    for c in range(x1.shape[3]):
                        batch_x_min[i,:,:,c] = x1[i,:,:,c] * weights[i] + x2[i,:,:,c] * (1 - weights[i])
                        #     print(x)
                y1, y2 = batch_y, batch_y[indxran]          
                batch_y_min = np.zeros(y1.shape)
            
                for i in range(batch_x.shape[0]):
                    batch_y_min[i] = y1[i] * weights[i] + y2[i] * (1 - weights[i])
            
                batch_x = np.concatenate((batch_x,batch_x_min), axis=0)
                batch_y = np.concatenate((batch_y,batch_y_min), axis=0)
                                   
            yield batch_x, batch_y

class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
