import tensorflow as tf
import numpy as np

from utils.loading import loadOmniglot
from utils.dataprep import partitionByClass, prepareBatch, invert_img, deinvert_img, resize_img, subtract_mean, augment_by_rotations
from utils.cnn6 import getPrototypes, getDistances, computeLoss, runModel, encoder_cnn_noaffine
from utils.visualize import plot_encoded_data, visualize

np.random.seed(123)

#train 0 = full, 1 = subset, 2 = subset
#limit is limiting the number of loaded images for faster prototyping
#for actual training, go for train = 0, limit = None, that will load the full training data
labels_train, images_train, info_train, labels_val, images_val, info_val = loadOmniglot(
    path = "data/",train = 1, limit = 400
    )

#resizing and inverting (so that the letter is 1 and background 0)
X_train = resize_img(images_train,105,105,28,28)
X_val = resize_img(images_val,105,105,28,28)

#switch to test how well the classifier does on partially damaged data
#the advantage of Gaussian prototypical network is in dealing with bad datapoints
damage = False
if damage == True:

    p = 0.75
    mask_train = (np.random.rand((X_train.shape[0])) < p).astype(int)
    mask_val = (np.random.rand((X_val.shape[0])) < p).astype(int)
    mask_train = mask_train.reshape([-1,1,1])
    mask_val = mask_val.reshape([-1,1,1])

    res = 7
    damage_fn = lambda I: resize_img(resize_img(I,28,28,res,res),res,res,28,28)
    X_train = mask_train*damage_fn(X_train) + (1-mask_train)*X_train
    X_val = mask_val*damage_fn(X_val) + (1-mask_val)*X_val

    noise = 0.0
    X_train = X_train + mask_train*np.random.normal(scale = noise, size = np.shape(X_train))
    X_val = X_val + mask_val*np.random.normal(scale = noise, size = np.shape(X_val))

    print("Data was purposefully damaged.")

X_train = invert_img(X_train)
X_val = invert_img(X_val)

y_train = labels_train
y_val = labels_val


augmentation = True #adds rotations by 90 degree multiples
if augmentation == True:
    X_train,y_train = augment_by_rotations(X_train,y_train, [0,1,2,3])
    X_val,y_val = augment_by_rotations(X_val,y_val, [0,1,2,3])


#subtracting mean
X_train, train_means = subtract_mean(X_train)
X_val, val_means = subtract_mean(X_val)

print("X_train",X_train.shape)
print("X_val",X_val.shape)

if damage == True:
    visualize(
        X_train[range(200),:,:],
        output = "damaged_visualization_test.png",
        width = 20
        )

embed_dim = 64 #small version
#embed_dim = 512

N_classes = 60 #number of classes at each batch during training
N_classes_val = 20 #number of clases at each batch during validation=test
N_support = 1 #number of support images per class = k of the k-shot classification
N_query = 19 #number of query images, on Omniglot N_query + N_support <= 20
dropout = 0.0
print_every = 1
plot_every = 100
val_every = 5 #in epochs

H = 28
W = 28

epochs = 30

sigma_mode_dict = {
0: "constant",
1: "radius",
2: "diagonal",
3: "full" #not fully implemented, as not very useful
}

sigma_mode = sigma_mode_dict[1]

#if None, then no restoration, otherwise loads saved model,
#!! NEED TO MANUALLY MODIFY STARTING EPOCH IN runModel FOR LEARNING RATE ANNEALING TO WORK!
restore_file = None
#restore_file = "../prototype_archive/testingbig_radius_512_lr2_norm_v7_take2_specialdamage1_nc60_nct20_ns1_e248_i64.ckpt"

#visualize(deprocess_img(X_batch_support), output = "test.png", width = 5)

#defining TF model
tf.reset_default_graph()

#placeholders
X_support_ph = tf.placeholder(tf.float32, [None, H, W])
X_query_ph = tf.placeholder(tf.float32, [None, H, W])
y_support_ph = tf.placeholder(tf.int64, [None])
y_query_ph = tf.placeholder(tf.int64, [None])
class_ids_ph = tf.placeholder(tf.int64, [N_classes])

learning_rate_ph = tf.placeholder(tf.float32) #for changing learning rate during training
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

encoder = encoder_cnn_noaffine #choosen from utils/cnn6.py

with tf.variable_scope("support") as support_scope:
    X_support_encoded, sigma_support = encoder(X_support_ph, 1.0 - dropout, True, embed_dim = embed_dim, sigma_mode = sigma_mode)

with tf.variable_scope(support_scope,reuse = True):
    X_query_encoded, sigma_query = encoder(X_query_ph, 1.0 - dropout, True, embed_dim = embed_dim, sigma_mode = sigma_mode)


#normalizing to a circle -- not useful
#X_support_encoded = X_support_encoded / tf.norm(X_support_encoded, axis = 1, keep_dims = True)
#X_query_encoded = X_query_encoded / tf.norm(X_query_encoded, axis = 1, keep_dims = True)

prototypes, devs = getPrototypes(X_support_encoded, y_support_ph, class_ids_ph, N_classes = N_classes, sigma_support = sigma_support, sigma_mode = sigma_mode)
print("prototypes",prototypes)

distances, distances_normed = getDistances(X_query_encoded, y_query_ph, prototypes, devs, class_ids_ph, sigma_mode = sigma_mode)
print("distances",distances)

loss, labels_predicted = computeLoss(y_query_ph, distances_normed, class_ids_ph, N_classes)
print("loss", loss)

optimizer = tf.train.AdamOptimizer(learning_rate_ph)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(loss)

#defining the TF session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Notes for running:
#-if not training, set training = None, other training = train_step
#-lr_halve sets how many epochs until learning rate is halved

#training = train_step
X_query_encoded_out = runModel(sess, labels_predicted, loss, class_ids_ph, X_train, y_train, X_support_ph, X_query_ph, y_support_ph, y_query_ph, is_training, keep_prob, learning_rate_ph, X_val, y_val, distances, dropout = dropout,
    epochs = epochs, N_classes = N_classes, N_classes_val = N_classes_val, N_support = N_support, N_query = N_query, print_every = print_every, plot_every = plot_every, val_every = val_every,
    training = train_step, X_query_encoded = X_query_encoded,
    checkpoint_name = "name_of_the_run", support_scope = support_scope, encoder = encoder, embed_dim = embed_dim, lr_start = 1e-3, lr_halve = 2000, sigma_support = sigma_support, sigma_mode = sigma_mode,
    sigma_query = sigma_query, restore_file = restore_file)

plot_encoded_data(X_query_encoded_out, None, None)
