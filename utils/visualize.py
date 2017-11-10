import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as image
import matplotlib.patches as patches

from sklearn.decomposition import PCA

def plot_encoded_data(X_encoded, y_predicted, y_true, N_classes, title = "", output = "clustering", X_raw = None, show_images = False, verbosity = 1, subset = None, prototypes = None, devs = None):

    plt.cla()
    plt.clf()
    plt.close()

    show_mid_characters = X_raw is not None

    #PCA on the vectors
    pca = PCA(n_components=2)
    pca.fit(X_encoded)
    PCA(copy=True, iterated_power='auto', n_components=2, random_state=123,svd_solver='auto', tol=0.0, whiten=False)

    if verbosity>0:
        print("Variation explained by PCA:",pca.explained_variance_ratio_)

    #reducing the data via the PCAed coordinates
    X_encoded_reduced = pca.transform(X_encoded)

    #calculating the sizes
    min0,max0 = np.min(X_encoded_reduced[:,0]), np.max(X_encoded_reduced[:,0])
    min1,max1 = np.min(X_encoded_reduced[:,1]), np.max(X_encoded_reduced[:,1])

    size0 = max0-min0
    size1 = max1-min1

    p = 0.1
    min0 -= size0*p
    max0 += size0*p
    min1 -= size1*p
    max1 += size1*p

    if subset is None:
        colors = cm.nipy_spectral(np.linspace(0, 1, N_classes))
        classes_used = np.unique(y_true)
    else:
        colors = cm.nipy_spectral(np.linspace(0, 1, np.min([N_classes,subset])))
        classes_used = np.random.choice(np.unique(y_true), size = np.min([N_classes,subset]), replace = False)


    if show_mid_characters:
        #calculating centers of clusters
        means = np.zeros((N_classes,2))
        N = np.shape(X_encoded)[0]

        for j,c in enumerate(classes_used):
            inds = [i for i in range(N) if y_true[i] == c]
            mean = np.mean(X_encoded_reduced[inds,:], axis = 0)
            means[j,:] = np.reshape(mean,(2))

        variance = np.sqrt(np.reshape(np.var(means, axis = 0),(2)))


    fig, ax = plt.subplots()
    plt.title(title + " true labels")
    N = np.shape(X_encoded)[0]
    handles_true = []

    ax.set_ylim((min1,max1))
    ax.set_xlim((min0,max0))

    for j,c in enumerate(classes_used):

        inds_true = [i for i in range(N) if y_true[i] == c]

        if ((show_mid_characters) and (not show_images)):

            image_to_show = X_raw[inds_true[0],:,:]

            plt.set_cmap('binary')
            img_size = 0.04
            ax.imshow(image_to_show, aspect='auto', extent=(means[j,0]-size0*img_size, means[j,0]+size0*img_size, means[j,1]-size1*img_size, means[j,1]+size1*img_size), zorder=-1)

            rect = patches.Rectangle((means[j,0]-size0*img_size,means[j,1]-size1*img_size),2*size0*img_size,2*size1*img_size,linewidth=1,edgecolor=colors[j],facecolor='none')
            ax.add_patch(rect)

        if prototypes is not None:

            prototypes_reduced = pca.transform(prototypes)

            plt.plot(prototypes_reduced[j,0],prototypes_reduced[j,1],color = colors[j], marker = "*", markersize=15)

        if ((prototypes is not None) and (devs is not None)):

            prototypes_reduced = pca.transform(prototypes)

            dev = devs[j]
            diag = np.diag(dev)

            diag_reduced = np.array(pca.transform(diag))
            lengths = np.linalg.norm(diag_reduced, axis = 1)
            majori,minori = np.argsort(-1.0*lengths)[0:2]

            major = np.linalg.norm(diag_reduced[majori])
            minor = np.linalg.norm(diag_reduced[minori])

            majora = np.arctan2(diag_reduced[majori,1],diag_reduced[majori,0])
            minora = np.arctan2(diag_reduced[minori,1],diag_reduced[minori,0])

            e = patches.Ellipse(xy = (prototypes_reduced[j,0],prototypes_reduced[j,1]), width = major, height = minor, angle = majora/2.0/np.pi*360.0, color = colors[j], alpha = 0.1)
            ax.add_patch(e)


        if not show_images:
            handles_true.append(ax.scatter(X_encoded_reduced[inds_true,0],X_encoded_reduced[inds_true,1],facecolors = colors[j], edgecolors = colors[j], label = str(classes_used[j]),marker = "o",alpha = 1.0 ))

        if show_images:
            for i in inds_true:
                plt.set_cmap('binary')
                img_size = 0.03
                ax.imshow(X_raw[i,:,:], aspect='auto', extent=(X_encoded_reduced[i,0]-size0*img_size, X_encoded_reduced[i,0]+size0*img_size,X_encoded_reduced[i,1]-size1*img_size, X_encoded_reduced[i,1]+size1*img_size), zorder=-1)
                rect = patches.Rectangle((X_encoded_reduced[i,0]-size0*img_size,X_encoded_reduced[i,1]-size1*img_size),2*size0*img_size,2*size1*img_size,linewidth=1,edgecolor=colors[j],facecolor='none')
                ax.add_patch(rect)


    plt.savefig(output+"_true.png", bbox_inches='tight', format='png', dpi=300)

    plt.cla()
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    plt.title(title + " pred labels")
    N = np.shape(X_encoded)[0]
    handles_pred = []

    ax.set_ylim((min1,max1))
    ax.set_xlim((min0,max0))

    for j,c in enumerate(classes_used):

        inds_pred = [i for i in range(N) if y_predicted[i] == c]

        if ((show_mid_characters) and (not show_images)):

            inds_true = [i for i in range(N) if y_true[i] == c]
            image_to_show = X_raw[inds_true[0],:,:]

            plt.set_cmap('binary')
            img_size = 0.04
            ax.imshow(image_to_show, aspect='auto', extent=(means[j,0]-size0*img_size, means[j,0]+size0*img_size, means[j,1]-size1*img_size, means[j,1]+size1*img_size), zorder=-1)

            rect = patches.Rectangle((means[j,0]-size0*img_size,means[j,1]-size1*img_size),2*size0*img_size,2*size1*img_size,linewidth=1,edgecolor=colors[j],facecolor='none')
            ax.add_patch(rect)

        if prototypes is not None:

            prototypes_reduced = pca.transform(prototypes)

            plt.plot(prototypes_reduced[j,0],prototypes_reduced[j,1],color = colors[j], marker = "*", markersize=15)

        if not show_images:
            handles_pred.append(ax.scatter(X_encoded_reduced[inds_pred,0],X_encoded_reduced[inds_pred,1],facecolors = colors[j], edgecolors = colors[j], label = str(classes_used[j]),marker = "o",alpha = 1.0))

        if show_images:
            for i in inds_pred:
                plt.set_cmap('binary')
                img_size = 0.03
                ax.imshow(X_raw[i,:,:], aspect='auto', extent=(X_encoded_reduced[i,0]-size0*img_size, X_encoded_reduced[i,0]+size0*img_size,X_encoded_reduced[i,1]-size1*img_size, X_encoded_reduced[i,1]+size1*img_size), zorder=-1)
                rect = patches.Rectangle((X_encoded_reduced[i,0]-size0*img_size,X_encoded_reduced[i,1]-size1*img_size),2*size0*img_size,2*size1*img_size,linewidth=1,edgecolor=colors[j],facecolor='none')
                ax.add_patch(rect)

    plt.savefig(output+"_pred.png", bbox_inches='tight', format='png', dpi=300)

    return None

def visualize(images, output = "characters.png", width = 20):

    images = 1.0 - images

    N,h,w = np.shape(images)

    Wi = int(np.floor(np.sqrt(N)))
    Wi = width

    Hi = int(np.ceil((1.0*N) / (1.0*Wi)))

    big = np.zeros((Hi*h,Wi*w))

    for yi in range(Hi):
        for xi in range(Wi):

            i = yi*Wi + xi
            if i < N:
                big[yi*h:(yi+1)*h,xi*w:(xi+1)*w] = images[i,:,:]

    print(big.shape)

    plt.axis('off')
    plt.set_cmap('Greys')
    plt.imshow(big)
    plt.savefig(output, bbox_inches='tight', format='png', dpi=1200)
    plt.show()
