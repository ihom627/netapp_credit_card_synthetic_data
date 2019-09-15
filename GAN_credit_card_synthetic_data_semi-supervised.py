import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers.core import Reshape, Dense, Dropout
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import to_categorical

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('random_dim', 10, """Number of dimention to generate data""")
tf.app.flags.DEFINE_integer('epochs', 5000, """Number of dimention to generate data""")
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of dimention to generate data""")
tf.app.flags.DEFINE_integer('gen_example', 600000, """Number of dimention to generate data""")
tf.app.flags.DEFINE_integer('example_per_cvs', 300000, """Number of dimention to generate data""")
tf.app.flags.DEFINE_string('output_dir', '/mnt/mount_0/dataset/creditcardfraud/gan/', """The folder to output dataset""")

def build_generator(latent_dim, data_dim):
    model = Sequential()

    model.add(Dense(16, input_dim=latent_dim))
    
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(32, input_dim=latent_dim))
    
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(data_dim, activation='tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    trans = model(noise)

    return Model(noise, trans)

def build_discriminator(data_dim, num_classes):
    model = Sequential()
    model.add(Dense(31, input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.25))
    model.add(Dense(16, input_dim=data_dim))
    model.add(LeakyReLU(alpha=0.2))
    
    model.summary()
    trans = Input(shape=(data_dim,))
    features = model(trans)
    valid = Dense(1, activation="sigmoid")(features)
    label = Dense(num_classes+1, activation="softmax")(features)
    return Model(trans, [valid, label])

def train(X_train,y_train,
          X_test,y_test,
          generator,discriminator,
          combined,
          num_classes,
          epochs, 
          batch_size):
    
    f1_progress = []
    half_batch = int(batch_size / 2)

    noise_until = epochs

    # Class weights:
    # To balance the difference in occurences of digit class labels.
    # 50% of labels that the discriminator trains on are 'fake'.
    # Weight = 1 / frequency
    cw1 = {0: 1/num_classes, 1: 1/num_classes}
    cw2 = {i: num_classes / half_batch for i in range(num_classes)}
    cw2[num_classes] = 1 / half_batch

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        trans = X_train[idx]

        # Sample noise and generate a half batch of new images
        noise = np.random.normal(0, 1, (half_batch, FLAGS.random_dim))
        gen_trans = generator.predict(noise)

        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        labels = to_categorical(y_train[idx], num_classes=num_classes+1)
        fake_labels = to_categorical(np.full((half_batch, 1), num_classes), num_classes=num_classes+1)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(trans, [valid, labels], class_weight=[cw1, cw2])
        d_loss_fake = discriminator.train_on_batch(gen_trans, [fake, fake_labels], class_weight=[cw1, cw2])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, FLAGS.random_dim))
        validity = np.ones((batch_size, 1))

        # Train the generator
        g_loss = combined.train_on_batch(noise, validity, class_weight=[cw1, cw2])

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))
        
        if epoch % 10 == 0:
            _,y_pred = discriminator.predict(X_test,batch_size=batch_size)
            #print(y_pred.shape)
            y_pred = np.argmax(y_pred[:,:-1],axis=1)
            
            f1 = f1_score(y_test,y_pred)
            print('Epoch: {}, F1: {:.5f}, F1P: {}'.format(epoch, f1, len(f1_progress)))
            f1_progress.append(f1)
            
    return f1_progress

def main():
    # Deterministic output.
    np.random.seed(10)
    tf.compat.v1.set_random_seed(20)

    df = pd.read_csv('/mnt/mount_0/dataset/creditcardfraud/creditcard.csv')

    df = df.drop('Time',axis=1)

    X = df.drop('Class',axis=1).values 
    y = df['Class'].values

    X -= X.min(axis=0)
    X /= X.max(axis=0)

    X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.1)

    

    ###################################
    # TensorFlow wizardry
    config = tf.ConfigProto()
    
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Create a session with the above options specified.
    tf.keras.backend.set_session(tf.Session(config=config))
    ###################################

    
    generator = build_generator(latent_dim=FLAGS.random_dim, data_dim=29)
    discriminator = build_discriminator(data_dim=29, num_classes=2)

    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
                                loss_weights=[0.5, 0.5],
                                optimizer=optimizer,
                                metrics=['accuracy'])

    noise = Input(shape=(FLAGS.random_dim,))
    trans = generator(noise)
    discriminator.trainable = False
    valid,_ = discriminator(trans)
    combined = Model(noise , valid)
    combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    rus = RandomUnderSampler(random_state=42)

    X_res, y_res = rus.fit_sample(X, y)

    X_res -= X_res.min()
    X_res /= X_res.max()

    X_test -= X_test.min()
    X_test /= X_test.max()

    X_test_res, y_test_res = rus.fit_sample(X_test,y_test)

    f1_p = train(X_res,y_res,
                    X_test,y_test,
                    generator,discriminator,
                    combined,
                    num_classes=2,
                    epochs=FLAGS.epochs, 
                    batch_size=FLAGS.batch_size)

    df_columns = ["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    new_df = pd.DataFrame(columns=df_columns)

    gen_trace_count = 0
    gen_file_index = 0
    collect_before_dump = 0
    while gen_trace_count < FLAGS.gen_example:
        gen_trace_size = FLAGS.batch_size if (FLAGS.gen_example - gen_trace_count) > FLAGS.batch_size else (FLAGS.gen_example - gen_trace_count)
        noise = np.random.normal(0, 1, (gen_trace_size, FLAGS.random_dim))
        gen_trans = generator.predict(noise)
        gen_results = discriminator.predict_on_batch(gen_trans)
            
        #print(tf.keras.backend.eval(gen_trans))
        #print(tf.keras.backend.eval(gen_trans)[0])
        #print(tf.keras.backend.eval(gen_results)[1])
        #print(tf.keras.backend.eval(gen_results)[1][1])
        #print(np.argmax(tf.keras.backend.eval(gen_results)[1][1]))
        #print(tf.keras.backend.eval(gen_results)[1][1][1])

        for index in range(gen_trace_size):
            if np.argmax(tf.keras.backend.eval(gen_results)[1][index]) < 2:
                accept_trans = tf.keras.backend.eval(gen_trans)[index]
                gen_trace_dict = {}
                for item in range(len(df_columns)-1):
                    gen_trace_dict[df_columns[item]] = accept_trans[item]
                gen_trace_dict["Class"] = str(np.argmax(tf.keras.backend.eval(gen_results)[1][index]))
                new_df = new_df.append(gen_trace_dict, ignore_index=True)
                gen_trace_count += 1
                collect_before_dump += 1

        print("Generating {}/{} transactions".format(gen_trace_count, FLAGS.gen_example))

        if collect_before_dump >= FLAGS.example_per_cvs or gen_trace_count >= FLAGS.gen_example:
            new_csv_filename = "creditcard_gan_{}.csv".format(gen_file_index)
            output_csv_filename = os.path.join(FLAGS.output_dir, new_csv_filename)
            print("Writing {}".format(output_csv_filename))
            new_df.to_csv(output_csv_filename)
            new_df = pd.DataFrame(columns=df_columns)
                
            collect_before_dump = 0
            gen_file_index += 1

if __name__ == "__main__":
    main()