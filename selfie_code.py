import os
import time
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Inspired by Karpathy's blog post: #
# http://karpathy.github.io/2015/10/25/selfie/ #

# Function to parse the image. #
def decode_img(img, img_width, img_height):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_width, img_height])

def process_path(file_path, label, img_width, img_height):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_width, img_height)
    return img, label

def process_test_path(file_path, img_width, img_height):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img, img_width, img_height)
  return img

# Build the model. #
def build_model(
    img_width, img_height, n_layers=12, 
    n_filters=8, p_drop=0.75, batch_norm=True):
    x_inputs = tf.keras.Input(
        shape=(img_height, img_width, 3), name='image_inputs')
    
    curr_inputs = x_inputs
    for n_block in range(n_layers):
        block_name = "block" + str(n_block+1) + "_"
        if n_block % 3 == 0:
            cnn_strides = (2, 2)
            if n_block > 0:
                n_filters = n_filters*2
        else:
            cnn_strides = (1, 1)
        
        # CNN Layer. #
        if n_block == 0:
            x_1 = layers.Conv2D(
                n_filters, (5, 5), strides=cnn_strides, 
                padding='same', name=block_name+"cnn_1")(curr_inputs)
        else:
            x_1 = layers.Conv2D(
                n_filters, (3, 3), strides=cnn_strides, 
                padding='same', name=block_name+"cnn_1")(curr_inputs)
        
        if batch_norm:
            x1b = layers.BatchNormalization()(x_1)
            x1o = layers.LeakyReLU()(x1b)
        else:
            x1o = layers.LeakyReLU()(x_1)
        
        x_2 = layers.Conv2D(
            n_filters, (3, 3), 
            padding='same', name=block_name+"cnn2")(x1o)
        x2b = layers.BatchNormalization()(x_2)
        x2o = layers.LeakyReLU()(x2b)
        
        if batch_norm:
            x2b = layers.BatchNormalization()(x_2)
            x2o = layers.LeakyReLU()(x2b)
        else:
            x2o = layers.LeakyReLU()(x_2)
        
        # Residual Layer. #
        if cnn_strides != (1, 1) and \
            curr_inputs.shape[-1] != n_filters:
            x_add = layers.Conv2D(
                n_filters, (3, 3), strides=cnn_strides, 
                padding="same", name=block_name+"residual")(curr_inputs)
        elif curr_inputs.shape[-1] != n_filters:
            x_add = layers.Conv2D(
                n_filters, (1, 1), 
                padding="same", name=block_name+"residual")(curr_inputs)
        elif cnn_strides != (1, 1):
            x_add = layers.Conv2D(
                n_filters, (3, 3), strides=cnn_strides, 
                padding="same", name=block_name+"residual")(curr_inputs)
        else:
            x_add = curr_inputs
        x_norm = layers.BatchNormalization()(x_add)
        x_relu = layers.LeakyReLU()(x_norm)
        
        x_residual  = x2o + x_relu
        curr_inputs = x_residual
    
    # Average Pooling. #
    x_avg_pool = layers.AveragePooling2D((2, 2))(x_residual)

    # Linear Layer. #
    x_flatten = layers.Flatten()(x_avg_pool)
    x_linear1 = layers.Dense(
        128, activation="relu", name="linear1")(x_flatten)
    x_linear2 = layers.Dense(
        32, activation="relu", name="linear2")(x_linear1)
    x_outputs = layers.Dense(
        1, activation="linear", name="outputs")(x_linear2)
    
    # Define the Keras model. #
    selfie_model = tf.keras.Model(inputs=x_inputs, outputs=x_outputs)
    return selfie_model

def model_loss(true_score, pred_score):
    loss_fn = tf.keras.losses.Huber(delta=0.05)
    return 100.0*loss_fn(true_score, pred_score)

# Compile into a Tensorflow function. #
@tf.function
def train_step(
    model, images, true_scores, 
    optimizer, learning_rate=1.0e-3):
    optimizer.lr.assign(learning_rate)
    
    with tf.GradientTape() as grad_tape:
        pred_scores = model(images, training=True)
        tmp_losses  = model_loss(true_scores, pred_scores)
        
        tmp_gradients = \
            grad_tape.gradient(tmp_losses, model.trainable_variables)
        optimizer.apply_gradients(
            zip(tmp_gradients, model.trainable_variables))
    return tmp_losses

def train(
    model, train_dataset, valid_dataset, 
    batch_size, epochs, width, height, optimizer, 
    init_lr=1.0e-3, decay=0.75, display_step=100):
    n_updates  = 0
    epoch_step = int(len(train_dataset) / batch_size)
    tmp_train_list = []
    
    tot_losses = 0.0
    for epoch in range(epochs):
        print("Epoch", str(epoch) + ":")
        start_time = time.time()
        learn_rate = decay**epoch * init_lr
        
        # Cool the GPU. #
        if (epoch+1) % 5 == 0:
            print("Cooling GPU.")
            time.sleep(120)
        
        for n_updates in range(epoch_step):
            batch_idx = \
                np.random.permutation(len(train_dataset))[:batch_size]
            batch_df  = train_dataset.iloc[batch_idx]
            
            image_batch = np.concatenate(
                tuple([np.expand_dims(np.array(
                    process_test_path(str(x), width, height)), axis=0) \
                        for x in list(batch_df["filename"].values)]), axis=0)
            image_label = tf.cast(batch_df["score"].values, tf.float32)
            
            tmp_losses = train_step(
                model, image_batch, image_label, 
                optimizer, learning_rate=learn_rate)
            tot_losses += tmp_losses.numpy()
            
            if (n_updates+1) % display_step == 0:
                tmp_string = "Intermediate Loss at Epoch " + str(epoch)
                tmp_string += " step " + str(n_updates+1) + ":"
                print(tmp_string, str(tot_losses/(n_updates+1)))
        
        tmp_true_scores = []
        tmp_pred_scores = []
        for image_batch, image_label in valid_dataset:
            tmp_scores = model.predict(image_batch).flatten()
            tmp_pred_scores.append(tmp_scores)
            tmp_true_scores.append(image_label.numpy())
        
        tmp_pred_scores = np.concatenate(tuple(tmp_pred_scores))
        tmp_true_scores = np.concatenate(tuple(tmp_true_scores))
        mae_error = np.mean(np.abs(tmp_true_scores - tmp_pred_scores))
        
        avg_losses = tot_losses / epoch_step
        tot_losses = 0.0
        elapsed_time = (time.time() - start_time) / 60.0
        tmp_train_list.append((epoch+1, avg_losses, mae_error))
        
        epoch += 1
        print("Learning Rate:", str(optimizer.lr.numpy()))
        print("Elapsed time:", str(elapsed_time), "mins.")
        print("Average Epoch Loss:", str(avg_losses) + ".")
        print("Validation Mean Abs. Error:", str(mae_error) + ".")
        print("-" * 75)
    
    tmp_train_df = pd.DataFrame(
        tmp_train_list, columns=["epoch", "train_huber_loss", "valid_mae"])
    return tmp_train_df

# Load the Selfie dataset. #
tmp_path = "C:/Users/admin/Desktop/Data/Selfie-dataset/"
tmp_selfie_df = pd.read_csv(tmp_path + "selfie_dataset_annotations.csv")

# The Selfie classifier task. #
label_dict = sorted(tmp_selfie_df.columns[2:])

# Split into Train and Validation data sets. #
percent_train = 0.7
percent_valid = 0.2

np.random.seed(1234)
idx_perm = np.random.permutation(len(tmp_selfie_df))

n_train  = int(percent_train * len(tmp_selfie_df))
n_valid  = n_train + int(percent_valid * len(tmp_selfie_df))
train_df = tmp_selfie_df.iloc[idx_perm[:n_train]]
valid_df = tmp_selfie_df.iloc[idx_perm[n_train:n_valid]]
test_df  = tmp_selfie_df.iloc[idx_perm[n_valid:]]

# Define the Neural Network. #
learning_rate = 1.0e-3
gradient_clip = 2.0

batch_norm   = False
img_height   = 150
img_width    = 150
n_layers     = 12
decay_rate   = 0.90
restore_flag = False

n_epochs   = 20
batch_size = 32
n_classes  = len(label_dict)

# Define the checkpoint callback function. #
train_loss = \
    "C:/Users/admin/Desktop/TF_Models/selfie_model/selfie_losses"
ckpt_path  = \
    "C:/Users/admin/Desktop/TF_Models/selfie_model/selfie_model"
ckpt_model = \
    "C:/Users/admin/Desktop/TF_Models/selfie_model/selfie_keras_model"

if batch_norm:
    test_loss  = \
        "C:/Users/admin/Desktop/TF_Models/selfie_model/selfie_test_losses.csv"
    train_loss += ".csv"
    ckpt_path  += ".ckpt"
else:
    test_loss  = "C:/Users/admin/Desktop/TF_Models/" +\
        "selfie_model/selfie_test_losses_no_batch_norm.csv"
    train_loss += "_no_batch_norm.csv"
    ckpt_path  += "_no_batch_norm.ckpt"
    ckpt_model += "_no_batch_norm"
ckpt_dir = os.path.dirname(ckpt_path)

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

if not restore_flag:
    try:
        os.mkdir(ckpt_model)
    except:
        shutil.rmtree(ckpt_model)
        os.mkdir(ckpt_model)

# Load the weights if continuing from a previous checkpoint. #
selfie_model = build_model(img_width, img_height, n_filters=16)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, clipnorm=2.0)

if restore_flag:
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, model=selfie_model)
    ck_manager = tf.train.CheckpointManager(
        checkpoint, directory=ckpt_model, max_to_keep=1)
    
    status = checkpoint.restore(ck_manager.latest_checkpoint)

# Save the model as an image file. #
#tf.keras.utils.plot_model(
#    selfie_model, to_file=tmp_path+"selfie_cnn_model.png")

# Print out the model summary. #
print(selfie_model.summary())
with open(tmp_path+"selfie_model_params.txt", "w") as tmp_file_open:
    selfie_model.summary(print_fn=lambda x: tmp_file_open.write(x + "\n"))
print("-" * 75)

# Generate the Data pipeline. #
# Shuffle was done within the train function as the Tensorflow #
# shuffle function took too long to shuffle the dataset.       #
one_valid_vec = np.ones(len(valid_df), dtype=np.int32)
valid_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(valid_df["filename"].values, tf.string),
    tf.cast(valid_df["score"].values, tf.float32), 
    img_width*one_valid_vec, img_height*one_valid_vec))
valid_dataset = valid_dataset.map(process_path)
valid_dataset = valid_dataset.batch(batch_size, drop_remainder=False)

print('Fit model on training data.')
tmp_train_df = train(
    selfie_model, train_df, valid_dataset, 
    batch_size, n_epochs, img_height, img_width, 
    optimizer, init_lr=learning_rate, decay=decay_rate)

# Evaluate on some random samples of the validation dataset. #
one_test_vec = np.ones(len(test_df), dtype=np.int32)
test_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(test_df["filename"].values, tf.string), 
    img_width*one_test_vec, img_height*one_test_vec))
test_dataset = test_dataset.map(process_test_path)
test_dataset = test_dataset.batch(batch_size, drop_remainder=False)

# Generate a DataFrame of predictions. #
test_true_scores = test_df["score"].values
test_pred_scores = []
for image_batch in test_dataset:
    tmp_scores = selfie_model.predict(image_batch).flatten()
    test_pred_scores.append(tmp_scores)
test_pred_scores = np.concatenate(tuple(test_pred_scores))

tmp_columns_df  = ["filename", "pred", "actual"]
test_results_df = pd.DataFrame([(
    test_df.iloc[x]["filename"], 
    test_pred_scores[x], test_true_scores[x]) \
        for x in range(len(test_df))], columns=tmp_columns_df)

test_mae = np.mean(np.abs(test_true_scores - test_pred_scores))
print("-"*50)
print("Test Set MAE:", str(test_mae))
print(test_results_df.head(25))
tmp_train_df.to_csv(train_loss, index=False)
test_results_df.to_csv(test_loss, index=False)