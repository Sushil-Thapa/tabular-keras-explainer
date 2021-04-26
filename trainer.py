import os
from models import get_model
from settings import SEED_VALUE,EPOCHS,BATCH_SIZE, model_checkpoint_path

os.environ['PYTHONHASHSEED']=str(SEED_VALUE)

import glob
import tensorflow as tf

tf.random.set_seed(SEED_VALUE)

from tensorflow.keras.optimizers import Adam


print("GPU Test:", tf.config.list_physical_devices('GPU'), tf.test.is_built_with_gpu_support(), tf.test.is_built_with_cuda())


def train(data, load_existing = True):
    print("Starting trainer module...")
    (X_train, y_train, X_val, y_val, datatype_val, input_shape) = data
    input_dim, output_dim = X_train.shape[1], y_train.shape[1] #n_feats, n_output
    model = get_model(input_dim, output_dim)

    if load_existing == True:
        print("Loading existing model from:", model_checkpoint_path)
        model.load_weights(model_checkpoint_path)

    else:
        print("Preparing to train the model..")
        # ## Training
        
        checkpoint_dir = os.path.dirname(model_checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_checkpoint_path,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        monitor='val_accuracy',
                                                        verbose=1)

        # es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        log_dir = "logs"
        exp_log_dir = len(os.listdir(log_dir)) + 1
        log_dir = f'{log_dir}/{exp_log_dir}'
        print(f"Saving models into: {log_dir}")

        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir= 3)


        optimizer = Adam(
            lr = 0.001,
            name = "Adam"
        )

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['categorical_crossentropy','accuracy'])

        history = model.fit(X_train, y_train, 
                            validation_data=(X_val, y_val), 
                            verbose=1, 
                            batch_size = BATCH_SIZE,
                            epochs=EPOCHS, 
                            shuffle=True,
                            callbacks = [cp_callback])
    
    return model
