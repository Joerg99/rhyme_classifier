#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pandas as pd
from tensorflow.contrib import learn
from input_helpers import InputHelper
from itertools import combinations





# Parameters
# ==================================================
def infer(batch_size_infer, x1_infer, x2_infer):
    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size", batch_size_infer, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
    tf.flags.DEFINE_string("eval_filepath", "validation_short.txt0", "Evaluate on this data (Default: None)")
    tf.flags.DEFINE_string("vocab_filepath", "runs/1543141697/checkpoints/vocab", "Load training time vocabulary (Default: None)") # setze vocab filepath ein
    tf.flags.DEFINE_string("model", "runs/1543141697/checkpoints/model-2000", "Load trained model checkpoint (Default: None)") # setze model filepath ein
    
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    
    
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")
    
    
    if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
        print("Eval or Vocab filepaths are empty.")
        exit()
    
    
    all_predictions = []
    for x1, x2 in zip(x1_infer, x2_infer):
        
        # load data and map id-transform based on training time vocabulary
        inpH = InputHelper()
        x1_test,x2_test = inpH.getTestDataSet_infer(x1, x2, FLAGS.vocab_filepath, 30)
        
        print("\nEvaluating...\n")
        
        # Evaluation
        # ==================================================
        checkpoint_file = FLAGS.model
        print checkpoint_file
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                sess.run(tf.initialize_all_variables())
                saver.restore(sess, checkpoint_file)
        
                # Get the placeholders from the graph by name
                input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
                input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
        
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/distance").outputs[0]
        
                accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        
                sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]
        
                #emb = graph.get_operation_by_name("embedding/W").outputs[0]
                #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
                # Generate batches for one epoch
                batches = inpH.batch_iter(list(zip(x1_test,x2_test)), 2*FLAGS.batch_size, 1, shuffle=False)
                # Collect the predictions here
                all_d=[]
                for db in batches:
                    x1_dev_b,x2_dev_b = zip(*db)
    #                 print('db ', db)
    #                 print('********************')
    #                 print('x1_dev_b')
    #                 print(x1_dev_b)
    #                 print('********************')
                    batch_predictions = sess.run([predictions], {input_x1: x1_dev_b, input_x2: x2_dev_b, dropout_keep_prob: 1.0})
        all_predictions.append(list(batch_predictions))
    return all_predictions



def read_training_data_and_tag_rhyme():
#     data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/chicago/stanza_wo_alits/train_stanza.txt', delimiter='\t' ,header=None, skip_blank_lines= False)
    data = pd.read_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/stanza_wo_alits/test_rhyme.txt', delimiter='\t' ,header=None, skip_blank_lines= False)
    print('read in')
    
    data[3] = ""  # column for rhyme
    #built quatrains as list for rhyme classifications
    verse = []
    quatrain = []
    all_quatrains = []
    for i in range(1, len(data[0])):
        if type(data.iat[i, 1]) != float:
            if data.iat[i,1] == 'newline':
                quatrain.append(verse)
                verse = []
            elif data.iat[i, 1] == 'sos':
                quatrain.append(verse)
                all_quatrains.append(quatrain)
                
                quatrain = []
                verse = []
            else:
                verse.append(data.iat[i, 1])
    quatrain.append(verse)
    all_quatrains.append(quatrain)
    print('made quatrains')
    print(len(all_quatrains))
    for q in all_quatrains:
        print(q)
    
    all_combis_quatrainwise = []
    for quatrain in all_quatrains:
        quatrain_combis = []
        for sent in quatrain:
            combis = list(combinations(sent, 2))
            quatrain_combis.extend(combis)
        all_combis_quatrainwise.append(quatrain_combis)
    
    x1 = []
    x2 = []
    for quatrain in all_combis_quatrainwise:
        temp_x1 = []
        temp_x2 = []
        for tup in quatrain:
            temp_x1.append(tup[0])
            temp_x2.append(tup[1])
        x1.append(temp_x1)
        x2.append(temp_x2)

    
    results = infer(300,x1, x2)
    all_ryhmes_counter = []
    for quatrain in results:
        counter = 0
        for result in quatrain:
            for value in result:
                if value < 0.3:
                    counter += 1
        all_ryhmes_counter.append(counter)
                 
    print(all_ryhmes_counter)
    
    j = -1
    for i in range(len(data)):
        if data.iat[i, 1] == 'sos':
            j+= 1
            data.iat[i, 3] = all_ryhmes_counter[j]
        else:
            data.iat[i, 3] = all_ryhmes_counter[j]
    
    print(data.head(30))
    with open('rhyme results', 'w') as file:
        for value in all_ryhmes_counter:
            file.write(str(value)+' ')
    
    data.to_csv('/home/joerg/workspace/emnlp2017-bilstm-cnn-crf/data/deepspeare/dev_stanza_rhyme.txt', sep='\t', columns=(0,1,2,3), header=None, index= False)





if __name__ == '__main__':
#     print(infer(3,[['house','mall', 'blah'], ['yes']], [['mouse', 'all', 'eeek'], ['yes']]))
    read_training_data_and_tag_rhyme()
