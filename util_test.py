import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
#from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib as mpl

plotmap = {7:'Bmp, Tgfb, Fgf' , 11: 'Wnt, Tgfb, Fgf' ,
    13: 'Wnt, Bmp, Fgf ',
    14: 'Wnt, Bmp, Tgfb' ,
    19: 'RA, Tgfb, Fgf ',
    21: 'RA, Bmp, Fgf' ,
    22: 'RA, Bmp, Tgfb ',
    25: 'RA, Wnt, Fgf' ,
    26: 'RA, Wnt, Tgfb' ,
    28: 'RA, Wnt, Bmp'}

singles = {1,2,4,8,16}
doubles = {3,5,9,17,6,10,18,12,20,24}
triples = {7,11,19,13,21,25,14,22,26,28}
quads = {30,29,27,23,15}
quints = {31}
target = {24,17,19,27,14}

all_train_corrs = []
all_train_factors = []
all_train_pc_corrs = []
all_test_corrs = []
all_test_factors = []
all_test_pc_corrs = []
all_preds_train = []
all_preds_test = []
accs_train = []
accs_test = []
pred_vote_train = None
pred_vote_test = None

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def weight_variable_uniform(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.random_uniform(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def max_pool(x, stride=2, filter_size=2):
    return tf.nn.max_pool(x, ksize=[1, filter_size, 1, 1],
                        strides=[1, stride, 1, 1], padding='VALID')

def unpool(value, stride = 2):

    sh = value.get_shape().as_list()
    out = tf.concat([value, tf.zeros_like(value)],2)
    out_size = [-1] + [stride * sh[1]] + sh[-2:]
    out = tf.reshape(out, out_size)
    return out

def cross_entropy(dec_out,enc_inp):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dec_out, labels=enc_inp))

def build_single_fc_layer(x_inp, Ws, bs):
    #h_fc = tf.nn.relu(tf.matmul(x_inp, Ws[0]) + bs[0])

    h_fc = tf.matmul(x_inp, Ws[0])

    return h_fc, 0

def build_single_inverse_fc_layer(y_out, Ws, bs):

    return tf.matmul(y_out , Ws[-3])

def build_fc_layer(x_inp, Ws, bs):

    h_fc = tf.matmul(x_inp, Ws[0]) + bs[0]

    h_fc = tf.nn.relu(h_fc)
    #return h_fc1
    #h_fc = tf.matmul(h_fc, Ws[1]) + bs[1]
    # moments = tf.nn.moments(h_fc,0,keep_dims=True)
    # gamma_0 = tf.Variable(1.)
    # beta_0 = tf.Variable(0.)
    #
    # h_fc = tf.nn.batch_normalization(h_fc,moments[0],moments[1],beta_0,gamma_0,0.00001)

    h_fc1 = tf.nn.relu(tf.matmul(h_fc, Ws[1]) + bs[1])

    # moments1 = tf.nn.moments(h_fc1, 0, keep_dims=True)
    # gamma_1 = tf.Variable(1.)
    # beta_1 = tf.Variable(0.)
    #
    # h_fc1 = tf.nn.batch_normalization(h_fc1,moments1[0],moments1[1],beta_1,gamma_1,0.00001)


    h_fc2 = tf.matmul(h_fc1,Ws[2]) + bs[2]
    #h_fc2 = tf.matmul(h_fc1,Ws[2]) + bs[2]


    return h_fc2 

def build_fc_layer_exp(x_inp, Ws, bs):
    h_fc = tf.nn.relu(tf.matmul(x_inp,Ws[0])+bs[0])
    switches = tf.sign(h_fc)
    h_fc_log = tf.multiply(switches,tf.log(tf.abs(h_fc)+0.01))
    return tf.matmul(h_fc_log,Ws[1])+bs[1] , switches

def build_inverse_fc_layer(y_out,Ws,bs):
    h_fc = tf.nn.relu(tf.matmul(y_out + bs[-1],Ws[-1]))

    h_fc = tf.nn.relu(tf.matmul(h_fc + bs[-2],Ws[-2]))
    #h_fc = tf.matmul(h_fc + bs[-2],Ws[-2])


    return tf.matmul(h_fc + bs[-3],Ws[-3])

def build_inverse_fc_layer_exp(y_out,Ws,bs):
    h_fc_log = tf.matmul(y_out + bs[1],Ws[1])
    switches = tf.sign(h_fc_log)
    h_fc = tf.nn.relu(tf.multiply(switches,tf.exp(h_fc_log)))
    return tf.matmul(h_fc + bs[0],Ws[0])

def shuffle_in_unison(a, b):
    permutation = np.random.permutation(len(a))
    shuffled_a = []
    shuffled_b = []
    for index in permutation:
        shuffled_a.append(a[index])
        shuffled_b.append(b[index])
    return shuffled_a, shuffled_b

def evaluate(val_latent,val_predicted,val_target,test_latent,test_predicted,test_target,seqs_file):

    #savename = seqs_file.split("/")[-2]

    plt.plot(val_target,val_predicted,'b*',alpha=0.4)
    plt.plot(test_target,test_predicted,'r*',alpha=0.4)
    plt.show()

    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')

    #plt.savefig(savename+".png")

    norms = []
    for i in range(val_latent.shape[0]):
        norms.append(np.linalg.norm(val_latent[i,:]) )

    plt.hist(norms)
    plt.show()
    #plt.savefig(savename+"_val_latent_norm.png")

    norms = []
    for i in range(test_latent.shape[0]):
        norms.append(np.linalg.norm(test_latent[i,:]) )

    plt.hist(norms)
    plt.show()
    #plt.savefig(savename+"_test_latent_norm.png")

def to_dec(arr,units_per_treat = 1):

    if units_per_treat > 1:
        new_arr = []
        for i in range(0,len(arr),units_per_treat):
            new_arr.append(arr[i])
        arr = new_arr

    digits = len(arr)

    num = 0
    for i in range(digits):
        num = num + arr[i] * (2 ** (digits-1-i))
    return num

def replicate_array(treatments, num_treats):

    arrs = []
    for i in range(treatments.shape[1]):
        arrs.append(np.hstack([np.expand_dims(treatments[:,i],0).T]*num_treats))
    return np.hstack(arrs)

def eval_aggregate_perf(dec_test_labels,dec_train_labels):

    global all_train_corrs
    global all_train_factors
    global all_train_pc_corrs
    global all_test_corrs
    global all_test_factors
    global all_test_pc_corrs
    global all_preds_train
    global all_preds_test
    global accs_test
    global accs_train
    global pred_vote_test
    global pred_vote_train

    plt.figure()
    plt.title('Distribution of Pearson correlation coefficients')
    plt.hist(all_train_corrs,bins=100,color='b',alpha=0.4,label='train',weights=np.zeros_like(all_train_corrs) + 1. / len(all_train_corrs))
    plt.hist(all_test_corrs,bins=100,color='g',alpha=0.4,label='test',weights=np.zeros_like(all_test_corrs) + 1. / len(all_test_corrs))
    plt.legend()
    plt.savefig('1.png')
    print("pearson corr stats")
    print((np.mean(all_train_corrs),np.std(all_train_corrs),np.mean(all_test_corrs),np.std(all_test_corrs)))

    plt.figure()
    plt.title('Distribution of L1 ratio losses')
    plt.hist(all_train_factors,bins=100,color='b',alpha=0.4,label='train',weights=np.zeros_like(all_train_factors) + 1. / len(all_train_factors))
    plt.hist(all_test_factors,bins=100,color='g',alpha=0.4,label='test',weights=np.zeros_like(all_test_factors) + 1. / len(all_test_factors))
    plt.legend()
    plt.savefig('2.png')
    print('L1 ratio loss stats')
    print((np.mean(all_train_factors),np.std(all_train_factors),np.mean(all_test_factors),np.std(all_test_factors)))

    all_train_pc_corrs = np.vstack(all_train_pc_corrs)
    means = np.mean(all_train_pc_corrs,axis=0)
    stds = np.std(all_train_pc_corrs,axis=0)
    plt.figure()
    plt.title('Bar chart of training correlation by PC')
    plt.bar(list(range(all_train_pc_corrs.shape[1])),height= means,yerr= stds)
    plt.savefig('3.png')

    all_preds_test = np.hstack(all_preds_test)
    means = np.mean(all_preds_test, axis = 1)
    stds = np.std(all_preds_test, axis = 1)
    plt.figure()
    plt.bar(np.arange(all_preds_test.shape[0]), height = means, yerr = stds)
    plt.title('Rank of true label L1 ratio among all classes test')
    plt.savefig('4.png')

    all_preds_train = np.hstack(all_preds_train)
    means = np.mean(all_preds_train, axis=1)
    stds = np.std(all_preds_train, axis=1)
    plt.figure()
    plt.bar(np.arange(all_preds_train.shape[0]), height=means, yerr=stds)
    plt.title('Rank of true label L1 ratio among all classes train')
    plt.savefig('5.png')

    all_test_pc_corrs = np.vstack(all_test_pc_corrs)
    means = np.mean(all_test_pc_corrs, axis=0)
    stds = np.std(all_test_pc_corrs, axis=0)
    plt.figure()
    plt.title('Bar chart of test correlation by PC')
    plt.bar(list(range(all_test_pc_corrs.shape[1])), height=means, yerr=stds)
    plt.savefig('6.png')

    print("Test Accuracy")
    print(np.mean(accs_test),np.std(accs_test))

    print("Train Accuracy")
    print(np.mean(accs_train), np.std(accs_train))

    acc = 0

    print(pred_vote_test[0:100])
    individual_performance = {i:[0,0] for i in target}

    for i in range(len(pred_vote_test)):
        if max(pred_vote_test[i], key=pred_vote_test[i].get) == dec_test_labels[i]:
            acc += 1
            individual_performance[dec_test_labels[i]][0] += 1
        individual_performance[dec_test_labels[i]][1] += 1

    for i in individual_performance:
        print(i,individual_performance[i][0]/individual_performance[i][1])
    acc /= len(dec_test_labels)
    print("Test Voting accuracy")
    print(acc)

    acc = 0

    print(pred_vote_train[0:100])
    individual_performance = {i: [0, 0] for i in (singles | doubles | quints | triples | quads) - target}

    for i in range(len(pred_vote_train)):
        if max(pred_vote_train[i], key=pred_vote_train[i].get) == dec_train_labels[i]:
            acc += 1
            individual_performance[dec_train_labels[i]][0] += 1
        individual_performance[dec_train_labels[i]][1] += 1

    for i in individual_performance:
        print(i, individual_performance[i][0] / individual_performance[i][1])
    acc /= len(dec_train_labels)
    print("Train Voting accuracy")
    print(acc)

    np.save('all_train_corrs',np.array(all_train_corrs))
    np.save('all_test_corrs',np.array(all_test_corrs))
    np.save('alL_train_l1',np.array(all_train_factors))
    np.save('all_test_l1',np.array(all_test_factors))
    np.save('all_train_pc_corrs',all_train_pc_corrs)
    np.save('all_test_pc_corrs',all_test_pc_corrs)
    np.save('all_preds_train',all_preds_train)
    np.save('all_preds_test',all_preds_test)
    np.save('accs_test',np.array(accs_test))
    np.save('accs_train',np.array(accs_train))

def eval_perf(train_recs, train_latent, train_data, train_labels, test_recs, test_latent, test_data, test_labels,test_set,units_per_treat):

    global all_train_corrs
    global all_train_factors
    global all_train_pc_corrs
    global all_test_corrs
    global all_test_factors
    global all_test_pc_corrs
    global all_preds_train
    global all_preds_test
    global accs_train
    global accs_test
    global pred_vote_train
    global pred_vote_test

    singles = {1, 2, 4, 8, 16}
    doubles = {3, 5, 9, 17, 6, 10, 18, 12, 20, 24}
    triples = {7, 11, 19, 13, 21, 25, 14, 22, 26, 28}
    quads = {30, 29, 27, 23, 15}
    quints = {31}

    #test_set = singles | doubles | triples | quads | {31}

    corrs = []
    factors = []

    dec_train_labels = []
    for i in range(len(train_labels)):
        dec_train_labels.append(to_dec(train_labels[i],units_per_treat=units_per_treat))
    dec_train_labels = np.array(dec_train_labels)

    for label in set(dec_train_labels):
        treatment_recs = np.mean(train_recs[dec_train_labels == label],axis = 0)
        treatment_data = np.mean(train_data[dec_train_labels == label], axis = 0)
        print(pearsonr(treatment_data,treatment_recs))

    #train_set = (singles | doubles | triples | quads | quints )- test_set
    all_set = (singles | doubles | triples | quads | quints )
    acc = 0
    corrs = []
    factors = []

    train_indices = {}
    freqs = np.zeros((len(all_set), 1))

    if pred_vote_train is None:
        pred_vote_train = [{j: 0 for j in all_set} for i in range(len(train_labels))]

    for j in all_set:
        index = np.array([int(i) for i in format(j, '05b')])
        train_indices[j] = index

    for i in range(len(train_latent)):

        corrs.append(np.dot(train_data[i],train_recs[i])/(np.linalg.norm(train_data[i])*np.linalg.norm(train_recs[i])))
        factors.append(np.dot(np.abs(train_latent[i]),train_labels[i])/(np.sum(np.abs(train_latent[i]))))

        if factors[-1] > 1.01:
            print(train_latent[i],train_labels[i])
            print(factors[-1])
            raise Exception

        dots = {}
        inds = {}

        for j in all_set:
            index = train_indices[j]
            index = replicate_array(np.expand_dims(index, 0), units_per_treat).squeeze()
            #dots[j] = np.dot(np.abs(train_latent[i]), index) / np.dot(np.abs(train_latent[i]), 1-index)
            dots[j] = np.dot(np.abs(train_latent[i]), index) / np.linalg.norm(index)
            inds[dots[j]] = j

        # print("PREDICTION")
        # print(to_dec(train_labels[i], units_per_treat=units_per_treat))
        # print(dots)

        keys = sorted(inds.keys())
        for k in range(len(keys)):
            if inds[keys[k]] == to_dec(train_labels[i], units_per_treat=units_per_treat):
                freqs[k] = freqs[k] + 1
                if k == len(keys) - 1:
                    acc += 1
                #     print("correct")
                #     print(to_dec(train_labels[i], units_per_treat=units_per_treat))
                #     print(inds[keys[-1]],keys[-1],inds[keys[-2]],keys[-2])
                # else:
                #     print("wrong")
                #     print(to_dec(train_labels[i], units_per_treat=units_per_treat))
                #     print([int(i) for i in format(inds[keys[-1]], '05b')],keys[-1],inds[keys[-2]],[int(i) for i in format(inds[keys[-2]], '05b')],keys[-2],train_latent[i])

        pred_vote_train[i][inds[keys[-1]]] += 1

    all_preds_train.append(freqs)
    acc /= len(train_labels)
    accs_train.append(acc)

    #plt.figure()
    #plt.bar(np.arange(len(train_set)), freqs)
    #plt.title('Rank Counts of True Label')
    #plt.show()

    all_train_corrs.extend(corrs)
    all_train_factors.extend(factors)

    # plt.figure()
    # plt.title('Distribution of training correlations reconstructed vs. sample')
    # plt.hist(corrs)
    # plt.show()
    #
    # plt.figure()
    # plt.title('Distribution of training factors reconstructed vs. sample')
    # plt.hist(factors)
    # plt.show()

    corrs = []
    for i in range(train_data.shape[1]):
        corrs.append(pearsonr(train_recs[:,i],train_data[:,i])[0])

    all_train_pc_corrs.append(np.array(corrs))

    # plt.figure()
    # plt.title('Bar chart of correlation by PC')
    # plt.bar(list(range(train_data.shape[1])),height=corrs)
    # plt.show()

    acc = 0
    corrs = []
    factors = []

    test_indices= {}
    freqs = np.zeros((len(all_set),1))

    if pred_vote_test is None:
        pred_vote_test = [{j:0 for j in all_set} for i in range(len(test_labels))]

    for j in all_set:
        index = np.array([int(i) for i in format(j,'05b')])
        test_indices[j] = index

    for i in range(len(test_labels)):

        corrs.append(np.dot(test_data[i],test_recs[i])/(np.linalg.norm(test_data[i])*np.linalg.norm(test_recs[i])))
        factors.append(np.dot(np.abs(test_latent[i]),test_labels[i])/np.sum(np.abs(test_latent[i])))

        dots = {}
        inds = {}

        for j in all_set:

            index = test_indices[j]
            index = replicate_array(np.expand_dims(index,0),units_per_treat).squeeze()
            #dots[j] = np.dot(np.abs(test_latent[i]),index) / np.dot(np.abs(test_latent[i]),1-index)
            dots[j] = np.dot(np.abs(test_latent[i]), index) / np.linalg.norm(index)
            inds[dots[j]] = j

        # print("PREDICTION")
        # print(test_labels[i])
        # print(dots)

        keys = sorted(inds.keys())
        for k in range(len(keys)):
            if inds[keys[k]] == to_dec(test_labels[i],units_per_treat=units_per_treat):
                freqs[k] = freqs[k] + 1
                if k == len(keys)-1:
                    acc += 1

        pred_vote_test[i][inds[keys[-1]]] += 1

        # if i % 100 == 0:
        #     print(test_latent[i])
        #     #print(test_recs[i])
        #     #print(test_data[i])
        #     print(corrs[-1])
        #     print(np.linalg.norm(test_data[i]))
        #     print(test_labels[i],dots)

    all_preds_test.append(freqs)
    all_test_corrs.extend(corrs)
    all_test_factors.extend(factors)

    corrs = []
    for i in range(test_data.shape[1]):
        corrs.append(pearsonr(test_recs[:, i], test_data[:, i])[0])

    all_test_pc_corrs.append(np.array(corrs))

    acc /= len(test_labels)
    accs_test.append(acc)

    # print(freqs)
    #plt.figure()
    #plt.bar(np.arange(len(test_set)),freqs)
    #plt.title('Rank Counts of True Label')
    #plt.show()
    #
    # plt.figure()
    # plt.title('Distribution of test correlations reconstructed vs. sample')
    # plt.hist(corrs)
    # plt.show()
    #
    # plt.figure()
    # plt.title('Distribution of test factors reconstructed vs. sample')
    # plt.hist(factors)
    # plt.show()

    return np.array(factors)

def eval_pca(pca,pca_test,seqs,labels,comp_gene_cols,conversion_str,comp_genes):

    comp_genes_xlabels = []
    flag = 0

    matrix1 = np.zeros((len(comp_genes), len(pca_test)))
    matrix2 = np.zeros((len(comp_genes), len(pca_test)))
    for label in pca_test:
        group = pca.inverse_transform(seqs[labels == label])
        group = np.mean(group, axis=0)
        matrix1[:, flag] = group[comp_genes]
        matrix2[:, flag] = means[label].squeeze()[comp_genes]
        comp_genes_xlabels.append(str([int(i) for i in format(label, conversion_str)]))
        flag += 1

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(matrix1, cmap='bwr', interpolation='nearest')
    ax[0].set_xticks(np.arange(0, matrix1.shape[0], 1.0))
    ax[0].set_yticks(np.arange(0, matrix1.shape[1], 1.0))

    ax[0].set_xticklabels(brian.comp_genes_xlabels)
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    ax[0].set_yticklabels(comp_gene_cols)

    cc = pearsonr(matrix1.flatten(), matrix2.flatten())[0]
    ax[0].set_title('Autoencoder cc:' + str(cc))

    ax[1].imshow(matrix2, cmap='bwr', interpolation='nearest')
    ax[1].set_xticks(np.arange(0, matrix2.shape[0], 1.0))
    ax[1].set_yticks(np.arange(0, matrix2.shape[1], 1.0))

    ax[1].set_xticklabels(brian.comp_genes_xlabels)
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    ax[1].set_yticklabels(comp_gene_cols)

    ax[1].set_title('Linear averaging cc:' + str(cc))

    plt.show()

def make_tSNE(X,y,target):

    manifold = TSNE(n_jobs=4)
    np.set_printoptions(suppress=True)
    manifold = manifold.fit_transform(X)

    fixed_y = []
    cdict = {}
    count = 0
    y = y.squeeze()
    z = y.copy()
    for i in range(len(y)):
        #print(y[i])
        if y[i] not in cdict:
            cdict[y[i]] = count
            count = count + 1
    for i in range(len(y)):
        fixed_y.append(cdict[y[i]])

    N = len(target) * 2  # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # define the data
    x = manifold[:,0]
    y = manifold[:,1]
    tag = np.array(fixed_y)  # Tag each point with a corresponding label

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, N, N + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    print(cdict)

    np.save('tsne-x',x)
    np.save('tsne-y',y)
    np.save('tag',tag)


    # make the scatter
    scat = ax.scatter(x, y, c=tag, cmap=cmap, norm=norm, alpha= 0.4)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
    cb.set_label('Custom cbar')
    ax.set_title('Discrete color mappings')
    plt.show()

    for treatment in target:

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        tag1 = tag.copy()
        #tag1[mask] = 0
        #print(treatment)
        #print((z!=treatment)&(z!=(treatment + 1000)))
        print(np.sum(tag1))
        # make the scatter
        scat = ax.scatter(x, y, c=tag1, cmap=cmap, norm=norm, alpha=0.01)
        scat = ax.scatter(x[(z==treatment)|(z==(treatment + 1000))], y[(z==treatment)|(z==(treatment + 1000))], c=tag1[(z==treatment)|(z==(treatment + 1000))], cmap=cmap, norm=norm, alpha=0.5)

        # create the colorbar
        cb = plt.colorbar(scat, spacing='proportional', ticks=bounds)
        cb.set_label('Custom cbar')
        ax.set_title(plotmap[treatment])
        plt.show()

def make_heatmap(comp_genes_matrix, control_genes_matrix, target_comp_genes_matrix,brian,comp_gene_cols):

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(comp_genes_matrix, cmap='bwr', interpolation='nearest')
    ax[0].set_xticks(np.arange(0, comp_genes_matrix.shape[1], 1.0))
    ax[0].set_yticks(np.arange(0, comp_genes_matrix.shape[0], 1.0))

    ax[0].set_xticklabels(brian.comp_genes_xlabels)
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    ax[0].set_yticklabels(comp_gene_cols)

    cc = pearsonr(comp_genes_matrix.flatten(), target_comp_genes_matrix.flatten())[0]
    ax[0].set_title('Autoencoder cc:' + str(cc))

    ax[1].imshow(control_genes_matrix, cmap='bwr', interpolation='nearest')
    ax[1].set_xticks(np.arange(0, comp_genes_matrix.shape[1], 1.0))
    ax[1].set_yticks(np.arange(0, comp_genes_matrix.shape[0], 1.0))

    ax[1].set_xticklabels(brian.comp_genes_xlabels)
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)
    ax[1].set_yticklabels(comp_gene_cols)

    cc = pearsonr(control_genes_matrix.flatten(), target_comp_genes_matrix.flatten())[0]
    ax[1].set_title('Linear averaging cc:' + str(cc))

    ax[2].imshow(target_comp_genes_matrix, cmap='bwr', interpolation='nearest')
    ax[2].set_xticks(np.arange(0, comp_genes_matrix.shape[1], 1.0))
    ax[2].set_yticks(np.arange(0, comp_genes_matrix.shape[0], 1.0))

    ax[2].set_xticklabels(brian.comp_genes_xlabels)
    for tick in ax[2].get_xticklabels():
        tick.set_rotation(90)
    ax[2].set_yticklabels(comp_gene_cols)
    ax[2].set_title('Target')
    plt.show()

def compare_distributions(samples1,samples2,units_per_treat,condition):

    # blue is sampled, green is target

    assert samples1.shape[1] == samples2.shape[1]

    n1 = samples1.shape[0]
    n2 = samples2.shape[0]
    d = samples1.shape[1]

    # univariate distr test

    fig, ax = plt.subplots(5,units_per_treat)
    plt.subplots_adjust(hspace=1.5)
    fig.suptitle('Latent unit activations for ' + plotmap[condition])

    for i in range(d):
        ax[i//units_per_treat, i%units_per_treat].hist(samples1[:,i],color='b',alpha=0.4,bins=20)
        ax[i//units_per_treat, i%units_per_treat].hist(samples2[:,i],color='g',alpha=0.4,bins=20)
        ax[i//units_per_treat, i%units_per_treat].set_title("unit " + str(i+1))

    plt.show()

    # bivariate distr test

    all_combos = []
    for i in range(d):
        for j in range(i+1,d):
            all_combos.append([i,j])

    all_combos = np.array(all_combos)
    plots = 5

    fig, ax = plt.subplots(plots,plots)
    plt.subplots_adjust(hspace=1.5)
    fig.suptitle('Sampled scatter plots for pairs of activations for ' + plotmap[condition])


    sampled_plots = np.random.choice(d*(d-1)//2,plots*plots,replace=False)
    sampled_combos = all_combos[sampled_plots]
    for i in range(sampled_combos.shape[0]):
        ax[i//plots,i%plots].scatter(samples1[:,sampled_combos[i,0]],samples1[:,sampled_combos[i,1]],c='b',alpha=0.4)
        ax[i//plots ,i%plots].scatter(samples2[:,sampled_combos[i,0]],samples2[:,sampled_combos[i,1]],c='g',alpha=0.4)
        ax[i // plots, i % plots].set_title(str(sampled_combos[i,0])+","+str(sampled_combos[i,1]))

    plt.show()

    # tsne test

    X = np.vstack([samples1,samples2])
    colors = ['b'] * n1 + ['g'] * n2

    manifold = TSNE(n_components=2)
    np.set_printoptions(suppress=True)
    manifold = manifold.fit_transform(X)

    plt.title('tSNE of sampled and real target latent distribution for ' + plotmap[condition])
    plt.scatter(manifold[0:n1,0],manifold[0:n1,1],c=['b'] * n1,label='sampled')
    plt.scatter(manifold[n1:,0],manifold[n1:,1],c=['g'] * n2,label='real')
    plt.legend(loc='upper right')

    plt.show()


