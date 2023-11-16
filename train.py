import os
import time
import datetime

import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from generator import generator
from discriminator import discriminator
from utils import load_image_train

# Change PATH variable to absolute/ relative path to the images directory on your machine which contains the train and val folders
PATH = 'data'

# Change these variables as per your need
EPOCHS = 100
BUFFER_SIZE = 9040
BATCH_SIZE = 4
LAMBDA = 100

class AutoCaCoNet(object):
    
    def __init__(self, checkpoint_dir='./Sketch2Color_training_checkpoints'):
        self.generator = generator()
        self.discriminator = discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)

        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.bce(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.bce(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.bce(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))
        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    @tf.function
    def train_step(self, input_image, target, epoch):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                              self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                         self.discriminator.trainable_variables))

    def fit(self, train_ds, epochs):
        for epoch in range(epochs):
            start = time.time()
            print("Epoch: ", epoch)

            for n, (input_image, target) in train_ds.enumerate():
                self.train_step(input_image, target, epoch)

            if (epoch + 1) % 5 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

def main():
    train_dataset = tf.data.Dataset.list_files(PATH+'/train/*.jpg')
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    autocaconet = AutoCaCoNet()    
    autocaconet.fit(train_dataset, EPOCHS)

if __name__ == '__main__':
    main()
