import tensorflow as tf
from pickle import load
from utils.models import build_model
from utils.classes import Progression

def generate_progression(build_dict, initial_chord = "4C_maj", tune_len = 32, top = 1, use_gpu = False):
    
    # Load model and vocabulary
    words_num2text = load(open("maps/words_num2text.txt",'rb'))
    words_text2num = load(open("maps/words_text2num.txt",'rb'))
    vocab_size = len(words_text2num)
    
    build_dict['batch_size'] = 1
    # Create model and loss function    
    model = build_model(**build_dict)
    
    model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    model.build(tf.TensorShape([1,None]))
    
    # Transform initial_chord to tensor (1 x 1)
    input_id = words_text2num[initial_chord]
    predictions = [input_id]
    
    for i in range(tune_len):
        model.reset_states()    
        input_eval = tf.expand_dims(predictions, 0)
        
        preds = tf.squeeze(model(input_eval), 0) # Returns (sequential, vocab_size)
        
        probs_top, idx_top = tf.math.top_k(preds[-1], top)
        
        logits_top = tf.expand_dims(tf.math.log(probs_top), 0)
        pred_id = idx_top[tf.random.categorical(logits_top, 1)[0,0].numpy()].numpy()
        
        predictions.append(pred_id)
        
        
    structure = [words_num2text[idx] for idx in predictions]

    return Progression(structure)