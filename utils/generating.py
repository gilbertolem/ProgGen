import tensorflow as tf
from pickle import load
from utils.models import build_model
from utils.classes import GenProgression


def generate_progression(initial_chord="4C_maj", tune_len=32, top=1, directory=None):
    
    # Load model and vocabulary
    words_num2text = load(open("maps/words_num2text.txt", 'rb'))
    words_text2num = load(open("maps/words_text2num.txt", 'rb'))

    if directory is None:
        directory = './training_checkpoints'
    
    build_dict = load(open(directory + '/build_dict', 'rb'))
    
    build_dict['batch_size'] = 1
    # Create model and loss function    
    model = build_model(**build_dict)
    
    model.load_weights(tf.train.latest_checkpoint(directory))
    model.build(tf.TensorShape([1, None]))
    
    # Transform initial_chord to tensor (1 x 1)
    input_id = words_text2num[initial_chord]
    predictions = [input_id]
    
    structure = [words_num2text[input_id]]

    print("\nGENERATING PROGRESSION...")

    current_length = 0
    while current_length < tune_len:
        model.reset_states()    
        input_eval = tf.expand_dims(predictions, 0)
        
        preds = tf.squeeze(model(input_eval), 0)  # Returns (sequential, vocab_size)
        
        probs_top, idx_top = tf.math.top_k(preds[-1], top)
        
        logits_top = tf.expand_dims(tf.math.log(probs_top), 0)
        pred_id = idx_top[tf.random.categorical(logits_top, 1)[0, 0].numpy()].numpy()
        pred_char = words_num2text[pred_id]
        
        if pred_id > 0:
            structure.append(pred_char)
            predictions.append(pred_id)

        current_length = GenProgression(structure).n_bars
        
    return GenProgression(structure)
