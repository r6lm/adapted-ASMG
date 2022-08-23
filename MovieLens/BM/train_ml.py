#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../pretrain")


# In[2]:


from __future__ import division
from __future__ import print_function
import os
from engine import *
from model import *
from utils import *

np.random.seed(1234)
tf.set_random_seed(123)


# In[3]:


tf.test.is_gpu_available()


# In[4]:


# load data to df
start_time = time.time()

data_df = pd.read_csv('../../datasets/ml_5yr_2014_2018_30seq.csv')
meta_df = pd.read_csv('../../datasets/ml_5yr_2014_2018_30seq_item_meta.csv')

data_df['itemSeq'] = data_df['itemSeq'].fillna('')  # empty seq are NaN
data_df['itemSeq'] = data_df['itemSeq'].apply(lambda x: [int(item) for item in x.split('#') if item != ''])
meta_df['cateId'] = meta_df['cateId'].apply(lambda x: [int(cate) for cate in x.split('#') if cate != ''])
meta_df = meta_df.sort_values(['itemId'], ascending=True).reset_index(drop=True)
cate_ls = meta_df['cateId'].tolist()

print('Done loading data! time elapsed: {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))))

num_users = data_df['userId'].max() + 1
num_items = data_df['itemId'].max() + 1
num_cates = max([max(i) for i in cate_ls]) + 1
cates, cate_lens = process_cate(cate_ls)


# In[5]:



train_config = {'method': 'BM',
#                 'dir_name': 'BM_train15-24_test25_10epoch',  # edit train test period range, number of epochs
                'dir_name': None, # varies across time
                'start_date': 20140101,  # overall train start date
                'end_date': 20181231,  # overall train end date
                'num_periods': 31,  # number of periods divided into
#                 'train_start_period': 15,
#                 'train_end_period': 24,
                'train_periods': 10, 
                'train_start_period': None, # varies across time
                'train_end_period': None, # varies across time
                'test_start_period': 25,
                'test_period': None, # varies across time
                'train_set_size': None,
                'test_set_size': None,

                'base_optimizer': 'adam',  # base model optimizer: adam, rmsprop, sgd
                'base_lr': None,  # base model learning rate
                'base_bs': 1024,  # base model batch size
                'base_num_epochs': 10,  # base model number of epochs
                'shuffle': True,  # whether to shuffle the dataset for each epoch
                }

EmbMLP_hyperparams = {'num_users': num_users,
                      'num_items': num_items,
                      'num_cates': num_cates,
                      'user_embed_dim': 8,
                      'item_embed_dim': 8,
                      'cate_embed_dim': 8,
                      'layers': [40, 20, 10, 1]
                      }


# In[6]:


data_df.info()


# In[7]:


data_df.head()


# In[8]:


# sort train data into periods based on num_periods: Each period has the same amount of observations
data_df = data_df[(data_df['date'] >= train_config['start_date']) & (data_df['date'] <= train_config['end_date'])]
data_df = data_df.sort_values(['timestamp']).reset_index(drop=True)
records_per_period = int(len(data_df) / train_config['num_periods'])
data_df['index'] = data_df.index
data_df['period'] = data_df['index'].apply(lambda x: int(x / records_per_period) + 1)


# In[9]:


data_df['period'].describe()


# In[10]:


data_df = data_df[data_df.period != train_config['num_periods'] + 1]  # delete last extra period at most one observation
period_df = data_df.groupby('period')['date'].agg(['count', 'min', 'max'])


# In[11]:


data_df["timestamp"].describe()


# In[12]:


period_df


# In[13]:


data_df = data_df.drop(['index', 'date', 'timestamp'], axis=1)


# In[14]:


data_df.tail()


# In[15]:


def train_base():

    # create an engine instance
    engine = Engine(sess, base_model)

    train_start_time = time.time()

    for epoch_id in range(1, train_config['base_num_epochs'] + 1):

        print('Training Base Model Epoch {} Start!'.format(epoch_id))

        base_loss_cur_avg = engine.base_train_an_epoch(epoch_id, train_set, train_config)
        print('Epoch {} Done! time elapsed: {}, base_loss_cur_avg {:.4f}'.format(
            epoch_id,
            time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time)),
            base_loss_cur_avg))

        test_auc, test_logloss = engine.test(test_set, train_config)
        print('test_auc {:.4f}, test_logloss {:.4f}'.format(
            test_auc,
            test_logloss))
        print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

        print('')

        # save checkpoint
        checkpoint_alias = 'Epoch{}_TestAUC{:.4f}_TestLOGLOSS{:.4f}.ckpt'.format(
            epoch_id,
            test_auc,
            test_logloss)
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_alias)
        saver.save(sess, checkpoint_path)
    
    # update test statistics
    test_aucs.append(test_auc)
    test_loglosses.append(test_logloss)


# # Modification

# In[ ]:


# initialize test statistics containers
test_aucs = []
test_loglosses = []

# range the test period 
for test_period in range(train_config["test_start_period"] + 1, train_config["num_periods"] + 1):
    
    # update train and test periods
    train_config['train_start_period'] = test_period - train_config["train_periods"]
    train_config['train_end_period'] = test_period - 1
    train_config['test_period'] = test_period
#     print("""train periods: {} - {}\ttest period: {}""".format(
#         train_start_period, train_end_period, test_period
#     ))
    train_config['dir_name'] = "BM_train{}-{}_test{}_10epoch".format(
        *map(lambda k: train_config[k], [
            "train_start_period", "train_end_period", "test_period"])
    )

#     print(train_config['base_lr'])

    

    orig_dir_name = train_config['dir_name']

    for base_lr in [1e-3]:

        print('')
        print('base_lr', base_lr)

        train_config['base_lr'] = base_lr

        train_config['dir_name'] = orig_dir_name + '_' + str(base_lr)
        print('dir_name: ', train_config['dir_name'])

        # create current and next set
        train_set = data_df[(data_df['period'] >= train_config['train_start_period']) &
                            (data_df['period'] <= train_config['train_end_period'])]
        test_set = data_df[data_df['period'] == train_config['test_period']]
        train_config['train_set_size'] = len(train_set)
        train_config['test_set_size'] = len(test_set)
        print('train set size', len(train_set), 'test set size', len(test_set))

        # checkpoints directory
        checkpoints_dir = os.path.join('ckpts', train_config['dir_name'])
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # write train_config to text file
        with open(os.path.join(checkpoints_dir, 'config.txt'), mode='w') as f:
            f.write('train_config: ' + str(train_config) + '\n')
            f.write('\n')
            f.write('EmbMLP_hyperparams: ' + str(EmbMLP_hyperparams) + '\n')
        
        
        
        # train and test
        
        # build base model computation graph
        tf.reset_default_graph()
        base_model = EmbMLP(cates, cate_lens, EmbMLP_hyperparams, train_config=train_config)
        
        with tf.Session() as sess:
            
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # create saver
            saver = tf.train.Saver(max_to_keep=80)
            
            train_base()
        
        # report results
        average_auc = sum(test_aucs) / len(test_aucs)
        average_logloss = sum(test_loglosses) / len(test_loglosses)
        print('test aucs', test_aucs)
        print('average auc', average_auc)
        print('')
        print('test loglosses', test_loglosses)
        print('average logloss', average_logloss)

        # write metrics to text file
        with open(os.path.join(checkpoints_dir, 'test_metrics.txt'), mode='w') as f:
            f.write('test_aucs: ' + str(test_aucs) + '\n')
            f.write('average_auc: ' + str(average_auc) + '\n')
            f.write('test_loglosses: ' + str(test_loglosses) + '\n')
            f.write('average_logloss: ' + str(average_logloss) + '\n')
        
        

