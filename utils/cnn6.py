import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.dataprep import partitionByClass, prepareBatch
from utils.visualize import plot_encoded_data, visualize
import math

def encoder_cnn_noaffine(X, keep_prob, is_training, embed_dim = 64, sigma_mode = "radius" ,H = 28, W = 28):
    #sigma_mode options:
    #sigma_mode = "radius" => a single real number per embedded point = confidence in all directions
    #sigma_mode = "diagonal" => embed_dim numbers per embedded point = diagonal of the covariance matrix

    #NOT FULLY IMPLEMENTED!!
    #sigma_mode = "full" => embed_dim*embed_dim numbers per embedded point = full covariance matrix

    if not sigma_mode in ["constant","radius","diagonal","full"]:
        print("The sigma mode "+str(sigma_mode)+" is not supported.")
        assert False

    sigma_dim_dictionary = {
    "constant": 0,
    "radius": 1,
    "diagonal": embed_dim,
    "full": int(embed_dim*(embed_dim+1)/2)
    }

    sigma_dim = sigma_dim_dictionary[sigma_mode] #dimension of the flattenned covariance matrix

    X_r = tf.reshape(X,[-1,H,W,1]) #adding the color channel (just add dimension)

    #CNN model specification
    filters = [3,3,3,3] #square filter sizes
    #num_filters = [128,256,512,embed_dim + sigma_dim]
    num_filters = [64,64,64,embed_dim + sigma_dim] #small version
    pools = [2,2,2,2]

    Ws_conv = []
    bs_conv = []

    #auxiliary variables
    a_conv = []
    a_conv_relu_normed = []
    a_conv_relu_normed_pad = []
    a_mixed = []
    a_pooled = []
    a_dropped = []

    previous_channels = 1 #starting number of channels
    a_previous = X_r #input to the first convolutional layer

    print("X",X_r)

    for i in range(len(filters)):

        filter_size = filters[i]
        num_filter = num_filters[i]
        pool = pools[i]

        Ws_conv.append(
            tf.get_variable("Wconv"+str(i), shape=[filter_size, filter_size, previous_channels, num_filter])
        )

        bs_conv.append(
            tf.get_variable("bconv"+str(i), shape=[num_filter])
        )

        #convolution
        a_conv.append(
            tf.nn.conv2d(a_previous, Ws_conv[i], strides=[1,1,1,1], padding='SAME') + bs_conv[i]
        )

        #ReLU
        if True:
            a_conv_relu_normed.append(
                tf.layers.batch_normalization( tf.nn.relu(a_conv[i]), axis=-1, training=is_training)
                )
        #testing bypassing the batch norm
        else:
            a_conv_relu_normed.append(
                tf.nn.relu(a_conv[i])
                )

        #2x2 Max Pooling layer with a stride of 2
        a_pooled.append(
            tf.nn.max_pool(a_conv_relu_normed[i], ksize = [1,pool,pool,1], strides=[1,pool,pool,1], padding = "VALID")
            )

        a_dropped.append( tf.nn.dropout(a_pooled[i], keep_prob) )

        previous_channels = num_filter
        a_previous = a_dropped[i]

        print("layer",i,a_previous)

    X_encoded = tf.reshape(a_previous,[-1,embed_dim + sigma_dim])

    if sigma_dim>0:
        X_encoded_positions, sigma_raw = tf.split(X_encoded, [embed_dim,sigma_dim], 1)

        #different versions of converting the raw sigma output to a proper covariance matrix

        sigma_transform_mode = "v7" #"v8","v9","v10"

        #v7 - basic stuff in a narrow range with softplus
        if sigma_transform_mode == "v7":
            offset = 1.0
            scale = 1.0
            sigma = offset + scale * tf.nn.softplus(sigma_raw)

        #v8 - basic stuff in a narrow range with sigmoid
        elif sigma_transform_mode == "v8":
            offset = 1.0
            scale = 1.0
            sigma = offset + scale * tf.sigmoid(sigma_raw)

        #v9 -- basic, wider range
        elif sigma_transform_mode == "v9":
            offset = 1.0
            scale = 4.0
            sigma = offset + scale * tf.sigmoid(sigma_raw)

        #v10 -- learnable - use it later in training by manually switching it on
        #since it has additional parameters, need to go to runModel and modify the restore_file loading
        elif sigma_transform_mode == "v10":

            init = tf.constant(1.0)

            scale = tf.get_variable("scale", initializer=init)
            div = tf.get_variable("div", initializer=init)
            offset = tf.get_variable("offset", initializer=init)

            sigma = offset + scale * tf.nn.softplus(sigma_raw/div)


        #set the covariance matrix to identity -- for testing
        #sigma = tf.ones_like(sigma)
    else:
        X_encoded_positions = X_encoded


    if sigma_mode == "full": #NOT FULL IMPLEMENTED AS NOT USEFUL
        assert False

    elif sigma_mode in ["radius","diagonal"]:
        sigma = sigma

    elif sigma_mode == "constant":
        sigma = tf.ones_like(X_encoded)

    #the sigma generated is the one from x^T Sigma x as a quadratic distance, i.e. the inverse of the usual notation
    print("sigma from encoder",sigma)
    return X_encoded_positions, sigma


def getPrototypes(X_support_encoded, y_support, class_ids, N_classes, sigma_support = 1.0, sigma_mode = "radius"):

    #X_encoded is concat of [positions,sigmas]

    means = []
    devs = []
    for c in tf.unstack(class_ids):

        mask = tf.expand_dims(tf.cast(tf.equal(y_support,c), tf.float32), axis = 1)

        if sigma_mode == "constant":
            sum1 = tf.reduce_sum( mask * X_support_encoded , axis = 0 )
            sum2 = tf.reduce_sum( mask * tf.ones_like(X_support_encoded), axis = 0)
            mean = sum1/sum2
            dev = tf.ones_like(mean)

        elif sigma_mode in ["radius","diagonal"]: #this should be mathematically correct, using sigma as 1/little sigma^2
            sum1 = tf.reduce_sum( mask * X_support_encoded * sigma_support , axis = 0 )
            sum2 = tf.reduce_sum( mask * tf.ones_like(X_support_encoded) * sigma_support , axis = 0)
            mean = sum1/sum2
            dev = sum2
        elif sigma_mode in ["full"]: #NOT FULLY IMPLEMENTED
            assert False

        means.append(mean)
        devs.append(dev)

    return means, devs

def getDistances(X_query_encoded, y_query, prototypes, devs, class_ids, sigma_mode):

    distances_list = []
    distances_normed_list = []

    for i in range(np.shape(tf.unstack(class_ids))[0]):

        p = prototypes[i]
        p = tf.reshape(p,[1,-1]) #to be able to substract it from all Xs at once
        diff = X_query_encoded - p
        dist = tf.pow( tf.norm(diff, axis = 1) , 2.0)

        if sigma_mode == "constant":
            dist_normed = tf.norm(diff, axis = 1)

        elif sigma_mode in ["radius","diagonal"]:
            sigma = tf.reshape(devs[i],[1,-1])

            diff_normed = tf.pow(diff,2.0) * sigma - tf.log(sigma) + 0.5*tf.log(2.0*np.pi)
            diff_normed = diff * tf.sqrt(sigma)
            dist_normed = tf.norm(diff_normed, axis = 1)

        elif sigma_mode == "full":
            assert False

        distances_list.append(dist)
        distances_normed_list.append(dist_normed)

    distances = tf.stack(distances_list,axis = 1) #[example, distance from that prototype]

    distances_normed = tf.stack(distances_normed_list,axis = 1) #[example, distance from that prototype]

    print("distances",distances)
    print("distances_normed",distances_normed)
    return distances, distances_normed


def calculateDistancesWithValues_fast(ps, devs, Xs, N_classes_val,sigma_mode):

    distances_list = []
    distances_normed_list = []

    for i in range(N_classes_val):

        p = ps[i]
        p = np.reshape(p,[1,-1])
        diff = Xs - p
        dist = np.power( np.linalg.norm(diff, axis = 1) , 2.0)
        distances_list.append(dist)

        if sigma_mode == "constant":
            dist_normed = np.linalg.norm(diff, axis = 1)

        elif sigma_mode in ["radius","diagonal"]:
            sigma = np.reshape(devs[i],[1,-1])

            diff_normed = diff * np.sqrt(sigma)
            dist_normed = np.linalg.norm(diff_normed, axis = 1)


        elif sigma_mode == "full":
            assert False

        distances_normed_list.append(dist_normed)

    distances = np.stack(distances_list, axis = 1)
    distances_normed = np.stack(distances_normed_list, axis = 1)

    return distances, distances_normed



def computeLoss(y_query, distances, class_ids, N_classes):#, N_query):

    logits = -1.0*distances

    local_class_ids = tf.argmin(distances, axis = 1)

    y_pred = tf.gather(class_ids, local_class_ids)

    labels = tf.zeros_like(y_query)
    for i,c in enumerate(tf.unstack(class_ids)):
        #print(i)
        mask = tf.expand_dims(tf.cast(tf.equal(y_query,c), tf.int64), axis = 1)
        mask = tf.reshape(mask, [-1])
        labels = labels + mask*(i % N_classes)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    loss = loss / (N_classes)

    loss = tf.reduce_mean(loss)

    return loss, y_pred

def runModel(session, y_predicted, loss, class_ids_ph, X_data, y_data, X_support_ph, X_query_ph, y_support_ph, y_query_ph, is_training, keep_prob, learning_rate_ph, X_val, y_val, distances, dropout = 0.0,
    epochs = 1, N_classes = 3, N_classes_val = 5, N_support = 5, N_query = 3, print_every = 100, plot_every = 1000, val_every = 1, training=None,X_query_encoded = None, checkpoint_name = "model", support_scope = None, encoder = None, embed_dim = 64, lr_start = 1e-3,
     lr_halve = 2000, sigma_support = 1.0, sigma_mode = "radius", sigma_query = None, restore_file = None):

    #just a hack for plotting
    terms_all = []

    #----------------------------------
    #defining validation variables
    with tf.variable_scope(support_scope,reuse = True):
        X_support_encoded_val, sigma_support_val = encoder(X_support_ph, 1.0 - dropout, True, embed_dim = embed_dim, sigma_mode = sigma_mode)

    with tf.variable_scope(support_scope,reuse = True):
        X_query_encoded_val, sigma_query_val = encoder(X_query_ph, 1.0 - dropout, True, embed_dim = embed_dim, sigma_mode = sigma_mode)

    prototypes_val, devs_val = getPrototypes(X_support_encoded_val, y_support_ph, class_ids_ph, N_classes = N_classes_val, sigma_support = sigma_support, sigma_mode = sigma_mode)
    #----------------------------------

    saver = tf.train.Saver(max_to_keep=None)

    if not restore_file is None: #if restoring from file

        if True:
            #loading only a subset of variables
            variables_all = tf.contrib.slim.get_variables_to_restore()

            def detectWords(what, where): return np.sum([int(w in where) for w in what]) > 0

            #to_omit = ["div","div:0","scale","scale:0","offset","offset:0"] #if going from v10 to something else
            to_omit = [] #otherwise

            variables_to_restore = [v for v in variables_all if not detectWords(to_omit,v.name.split('/'))]

            saver = tf.train.Saver(variables_to_restore) #if only a subset
            #saver = tf.train.Saver() #if want to load all

        saver.restore(session, restore_file)
        print("Model restored from"+restore_file+".")

    saver = tf.train.Saver() #to save all variables

    correct_predictions = tf.equal(y_predicted, y_query_ph)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    #needed for good batch generation
    ids_by_class_train = partitionByClass(X_data,y_data)

    training_now = training is not None

    if training_now:
        variables = [loss,correct_predictions,accuracy,y_predicted,X_query_encoded,sigma_query, training]
    else:
        variables = [loss,correct_predictions,accuracy,y_predicted,X_query_encoded, sigma_query]

    cnt = 0
    for e in range(epochs): #if starting from a restore file, set range(start_epoch,epochs)

        epoch_loss = 0.0
        epoch_acc = 0.0

        batch_size = (N_support + N_query)*N_classes

        for i in range(int(math.ceil(X_data.shape[0]/batch_size))): #going over the whole dataset
        #for i in range(1): #just for testing on already trained models

            X_batch_support, y_batch_support, X_batch_query, y_batch_query, class_ids = prepareBatch(X_data,y_data,ids_by_class_train,
                N_classes = N_classes, N_support = N_support, N_query = N_query)

            #visualize(deprocess_img(X_batch_support), output = "test.png", width = 5)

            lr = lr_start / np.power(2.0,(np.floor(cnt / lr_halve)))

            #print("lr = ",lr)

            feed_dict = {X_support_ph: X_batch_support,
                         X_query_ph: X_batch_query,
                         y_support_ph: y_batch_support,
                         y_query_ph: y_batch_query,
                         is_training: training_now,
                         keep_prob: 1.0 - dropout,
                         class_ids_ph: class_ids,
                         learning_rate_ph: lr
                        }

            if training_now:
                loss_out, correct_predictions_out, accuracy_out, y_predicted_out, X_query_encoded_out, sigma_query_out, _ = session.run(variables,feed_dict=feed_dict)
            else:
                loss_out, correct_predictions_out, accuracy_out, y_predicted_out, X_query_encoded_out, sigma_query_out = session.run(variables,feed_dict=feed_dict)

            epoch_loss += loss_out
            epoch_acc += accuracy_out

            if (cnt % print_every) == 0:
                print("# epoch="+str(e)+" it="+str(i)+"/"+str(int(math.ceil(X_data.shape[0]/batch_size)))+" loss="+str(loss_out)+" train acc="+str(accuracy_out))

                #hack for plotting
                if sigma_mode in ["radius","diagonal"]: #matrices embed_dim x embed_dim
                    if True:
                        terms_all += list(sigma_query_out.flatten())

                        thefile = open('sigmas_test.txt', 'w')
                        for cislo in terms_all:
                            thefile.write(str(cislo)+"\n")
                        thefile.close()

                        #print("all terms", len(terms_all))

            if (cnt % plot_every) == 0:

                showing_sigmas = False #want to visualize sigmas?

                #looking at distribution of sigmas ------------------
                sigmas = sigma_query_out
                #print(np.shape(sigmas))

                if showing_sigmas and sigma_mode == "full": #matrices embed_dim x embed_dim
                    assert False

                    diagonals = (np.diagonal(sigmas,axis1 = 1, axis2 = 2)).flatten()
                    mask_one = np.invert(np.eye(embed_dim) == 1)
                    mask = np.tile(mask_one,[np.shape(sigmas)[0],1,1])
                    off_diagonals = sigmas[mask]

                    plt.cla()
                    plt.clf()
                    plt.close()

                    plt.title("Off-diagonal terms of $\Sigma$s")
                    plt.hist(off_diagonals, bins=30)  # arguments are passed to np.histogram
                    filename = "tmp_imgs/"+str(checkpoint_name)+sigma_mode+"_sigma_offdiagonal_"+str(e)+"_i"+str(i)
                    plt.savefig(filename+".png", bbox_inches='tight', format='png', dpi=300)
                    #plt.show()

                    plt.cla()
                    plt.clf()
                    plt.close()

                    plt.title("Diagonal terms of $\Sigma$s")
                    plt.hist(diagonals, bins=30)  # arguments are passed to np.histogram
                    filename = "tmp_imgs/"+str(checkpoint_name)+sigma_mode+"_sigma_diagonal_"+str(e)+"_i"+str(i)
                    plt.savefig(filename+".png", bbox_inches='tight', format='png', dpi=300)
                    #plt.show()

                if showing_sigmas and sigma_mode in ["radius","diagonal"]: #matrices embed_dim x embed_dim

                    terms = sigmas.flatten()
                    print("sigmas",sigmas)

                    plt.cla()
                    plt.clf()
                    plt.close()

                    plt.title("Terms of $\Sigma$s")
                    plt.hist(terms, bins=30)  # arguments are passed to np.histogram
                    filename = "tmp_imgs/"+str(checkpoint_name)+sigma_mode+"_sigmas_"+str(e)+"_i"+str(i)
                    plt.savefig(filename+".png", bbox_inches='tight', format='png', dpi=300)
                    #plt.show()


                #plotting ---------------------
                title = "Nc="+str(N_classes)+" Ns="+str(N_support)+" Nq="+str(N_query)+" epoch="+str(e)+" acc="+str(accuracy_out)

                filename = "tmp_imgs/"+str(checkpoint_name)+"_chars_e"+str(e)+"_i"+str(i)
                plot_encoded_data(X_query_encoded_out, y_predicted_out, y_batch_query, N_classes, title, filename, X_raw = X_batch_query, show_images = True, subset = 10)

                filename = "tmp_imgs/"+str(checkpoint_name)+"_dots_e"+str(e)+"_i"+str(i)
                plot_encoded_data(X_query_encoded_out, y_predicted_out, y_batch_query, N_classes, title, filename, X_raw = X_batch_query, show_images = False, subset = 10)

            cnt += 1

        its = int(math.ceil(X_data.shape[0]/batch_size))
        epoch_loss = epoch_loss / its
        epoch_acc = epoch_acc / its
        print("Epoch = "+str(e)+" done: loss = "+str(epoch_loss)+" train acc = "+str(epoch_acc))


        #------------------------------------

        if (e % val_every) == 0: #doing validation

            variables_val = [X_query_encoded_val,prototypes_val, devs_val]

            ids_by_class_val = partitionByClass(X_val,y_val)

            max_shot = 20
            accs = [[] for i in np.arange(0,max_shot,1)]

            for shot in np.arange(1,max_shot,1):

                batch_size_val = (20)*N_classes_val
                reps = int(math.ceil(X_val.shape[0]/batch_size_val))

                #my hack for testing only on 1 and 5 shot to speed things up
                # if shot in [1,5]:
                #     q = reps
                # else:
                #     q = 0

                q = 5 #for fast validation during prototyping
                #q = reps #during testing for full validation set

                for r in range(q):

                    #print("shot="+str(shot)+" "+str(r)+"/"+str(int(reps)))

                    X_batch_support, y_batch_support, X_batch_query, y_batch_query, class_ids_val = prepareBatch(X_val,y_val,ids_by_class_val,
                        N_classes = N_classes_val, N_support = shot, N_query = 20-shot)

                    feed_dict_val = {X_support_ph: X_batch_support,
                                 X_query_ph: X_batch_query,
                                 y_support_ph: y_batch_support,
                                 y_query_ph: y_batch_query,
                                 is_training: False,
                                 keep_prob: 1.0,
                                 class_ids_ph: np.reshape((np.tile(class_ids_val, N_classes))[0:N_classes],[-1]),
                                 learning_rate_ph: 0.0 #doesn't matter at all, no step here in validation
                                }

                    X_query_encoded_out, prototypes_val_out, devs_val_out = session.run(variables_val,feed_dict=feed_dict_val)

                    distances_val_out, distances_normed_val_out = calculateDistancesWithValues_fast(prototypes_val_out,devs_val_out,X_query_encoded_out, N_classes_val, sigma_mode)

                    args = np.reshape(np.argmin(distances_normed_val_out, axis = 1),[-1]).astype(int)
                    y_predicted_out = class_ids_val[args]

                    accuracy_out_repaired = np.mean(y_predicted_out == y_batch_query)
                    accs[shot].append(accuracy_out_repaired)

            accs[0] = list(np.array(accs[1]) * 0.0) #hack to do 0 shot (tho not actually doing it)
            #print("accs",accs)

            val_accuracies = [np.mean(a) for a in accs]
            val_errorbars = [np.sqrt(np.var(a)) for a in accs]
            print("Validation:")
            for shot in np.arange(0,max_shot,1):
                print("Epoch = "+str(e)+" shot = "+str(shot)+" val acc = "+str(val_accuracies[shot])+" +- "+str(val_errorbars[shot]))

        #--------------------------------------

        #saving the model after each epoch
        save_path = saver.save(session, "model_checkpoints/"+checkpoint_name+"_e"+str(e)+"_i"+str(i)+".ckpt")
        #print("Model saved to "+str(save_path))

    return X_query_encoded_out
