from util_test import *
import logging
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import skew

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pickle
import sys

# evaluate PCA performance
EVAL_PCA = False
# MAR for marginalization baseline, LIN for linear baseline, refer to paper
BASELINE_METHOD = "MAR"
# samples the posterior to generate samples instead of taking the mean, not recommended
SAMPLE_GENERATOR = False
# evaluates the recosntruction and regularization performance of the autoencoder
EVAL_PERF = False
# how many epochs we are going to train only the prediction module
EPOCHS_ONLY_PRED = 0
# present analysis on complementarily regulated genes
COMP_GENES_ANALYSIS = False

data = pd.read_csv("undir_r1.concat.csv")
data = data.set_index('Unnamed: 0')

all_results = []
all_comp_gene_matrices = []
all_control_matrices = []
all_target_matrices = []

all_ae_profiles = []
all_control_profiles = []
all_target_profiles = []

all_generated_profiles = []
all_generated_labels = []

# The names of the complemntary regulated genes
# these genes will get special analysis treatment and can print heat map

#comp_gene_cols = [Hoxa1,Lefty1,Mixl1]
#comp_gene_cols = ['Dpysl4','Scube2','Prss43','Serpinf1','Fhl2','Lefty1','Lefty2','Pycr2','Cda','Tal2','Acss1','Cplx2','Lrrc4']
#comp_gene_cols = ['Stmn4','Tmem591','Scn1b','Tsku','Dusp14','Hoxb5os','Mreg','Pam','Matn4','S100a7a','Adamts5','Gprc5a']
comp_gene_cols = ['Lgals3','Dll3','Arl4d','Fam212a','Hoxb1','Wnt3a','Gm13715','Ptgs1','Scn9a','Vim','Saxo1','Chst7','Cited1'
                  ,'Qrfpr','Sprr2a3','Cdh2','Pcdhb2','T','Fst','Smc6','Evx1','Evx1os']
#comp_gene_cols = ['Arf4','Dach1','Rarb','Ckap4','Tead2','Ccnb2','Fam64a','H2afv','Nsg2','Pbx1','Cenpa','Irs4','Mex3a','Hoxc4','Hoxc5',
#                  'Ptprcap','Stk32a','Col1a2','Grifin','Igf2bp3']

# whether or not to upsample the data, highly recommended

sampling = True


num_treats = 5
conversion_str = '0' + str(num_treats) + 'b'

# important: conditions mapping key start from 1, if we don't have data concerning no treatment
# example conditions values when num_treats = 5
# {1: {1, 2, 4, 8, 16},
#  2: {3, 5, 6, 9, 10, 12, 17, 18, 20, 24},
#  3: {7, 11, 13, 14, 19, 21, 22, 25, 26, 28},
#  4: {15, 23, 27, 29, 30},
#  5: {31}}

conditions = {i:set() for i in range(1,num_treats + 1)}
all_conditions = set()
for i in range(1,2**num_treats):
    conditions[sum([int(i) for i in format(i,conversion_str)])].add(i)
    all_conditions.add(i)

# sets the test target condition group, this can be a set of numbers or simply an element of conditions
target = conditions[3]

bump = []
for i in range(2**num_treats):
    bump.append([int(i) for i in format(i,conversion_str)])
bump = np.array(bump)

# tcs: key: condition A, value: the list of conditions whose mean profiles will be averaged to
# predict the mean profile for condition A
tcs = {}
for label in all_conditions:
    if BASELINE_METHOD == "MAR":
        bin = [int(i) for i in format(label,conversion_str)]
        a = singles | doubles | triples | quads | quints #| sexts | septs
        for i in range(len(bin)):
            if bin[i] != 0:
                b = set(np.nonzero(bump[:,i].squeeze())[0])
                a = a & b
        tcs[label] = a - target
    elif BASELINE_METHOD == "LIN":
        bin = [int(i) for i in format(label, conversion_str)]
        bin.reverse()
        tcs[label] = set()
        for i in range(len(bin)):
            if bin[i] != 0:
                tcs[label].add(2 ** i)
    else:
        print("Baseline method not defined!")
        exit()

class Model():

    def __init__(self, latent_size, len, num_treats, filters, units_per_treat):

        assert latent_size >= num_treats * units_per_treat
        if EVAL_PCA:
            self.pca_ratios = []

        self.latent_size = latent_size
        self.len = len
        self.num_treats = num_treats
        self.units_per_treat = units_per_treat
        self.encoder_filters = filters

        self.expressions = tf.placeholder(tf.float32,shape=(None,self.len))
        self.treatments = tf.placeholder(tf.float32,shape=(None,num_treats * units_per_treat))


        self.learning_rate = 0.001

        self.encoder_filter_weights = [weight_variable([self.len,self.encoder_filters]), weight_variable([self.encoder_filters,self.encoder_filters]),weight_variable([self.encoder_filters,self.latent_size]) ]
        self.encoder_filter_biases = [weight_variable([self.encoder_filters]), weight_variable([self.encoder_filters]), weight_variable([self.latent_size])]


        self.decoder_filter_weights = [weight_variable([self.encoder_filters,self.len*2]),weight_variable([self.encoder_filters,self.encoder_filters]),weight_variable([self.latent_size,self.encoder_filters])]

        self.decoder_filter_biases = [weight_variable([self.encoder_filters]),weight_variable([self.encoder_filters]), weight_variable([self.latent_size])]


        self.latent_representation, self.decode_representation = self.make_encoding(self.expressions)
        self.dec_out = self.make_decoding(self.decode_representation)
        self.dec_out_mean = self.dec_out[:,:self.len]
        self.dec_out_log_sigma = tf.nn.relu(self.dec_out[:,self.len:] + 10) - 10

        self.target_decode_representation = tf.placeholder(tf.float32,shape=(None,self.latent_size))

        self.gen_out = self.make_decoding(self.target_decode_representation)
        self.gen_out_mean = self.gen_out[:,:self.len]
        self.gen_out_log_sigma = tf.nn.relu(self.gen_out[:,self.len:] + 10) - 10

        self.rec_loss  = tf.reduce_sum(tf.reduce_sum(  0.5 * self.dec_out_log_sigma
                                                         + tf.div(0.5 * tf.square(self.expressions - self.dec_out_mean)
                                                         , tf.exp(self.dec_out_log_sigma))))

        self.sum2 = tf.reduce_sum(tf.abs(tf.multiply(self.latent_representation,self.treatments)),axis=1)
        self.sum1 = tf.reduce_sum(tf.abs(tf.multiply(self.latent_representation,1-self.treatments)),axis=1)

        self.norm1 = tf.norm(self.latent_representation, axis=1)
        self.norm2 = tf.norm(tf.multiply(self.latent_representation, self.treatments), axis=1)

        self.group_norms = []
        for i in range(self.num_treats):
            self.group_norms.append(tf.norm(self.latent_representation[:,i * self.units_per_treat:(i+1)* self.units_per_treat],axis=1))
        self.group_lasso_loss = self.group_norms[0]
        for i in range(1,self.num_treats):
            self.group_lasso_loss += self.group_norms[i]

        self.proxy_target = tf.reduce_sum(tf.abs(self.latent_representation),axis=1)

        self.l1_loss = tf.reduce_sum(tf.divide(self.sum1,self.sum2))
        self.l2_loss = tf.reduce_sum(self.norm2)

        self.pred_loss = tf.reduce_sum(tf.divide(self.group_lasso_loss, self.norm1)) #self.l2_loss + self.l1_loss
        #self.pred_loss = tf.reduce_sum(self.group_lasso_loss) #self.l2_loss + self.l1_loss

        # another option
        #self.pred_loss = tf.reduce_sum(tf.norm(tf.multiply(self.latent_representation,self.inverse_treatments),axis = 1))

        self.joint_loss = self.rec_loss + 10 * self.pred_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.gvs_joint = self.optimizer.compute_gradients(self.joint_loss)
        self.gvs_pred = self.optimizer.compute_gradients(self.pred_loss)
        self.capped_joint = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in self.gvs_joint if grad is not None]
        self.capped_pred = [(tf.clip_by_value(grad, -0.1, 0.1),var) for grad, var in self.gvs_pred if grad is not None]
        self.train_joint = self.optimizer.apply_gradients(self.capped_joint)  # joint training.
        self.train_pred = self.optimizer.apply_gradients(self.capped_pred)

        self.sess = tf.InteractiveSession()  # OR: self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.initialize_all_variables())  # Intialize.
        self.saver = tf.train.Saver()


    def make_encoding(self,expressions):

        decode_representation = build_fc_layer(expressions,self.encoder_filter_weights,self.encoder_filter_biases)

        latent_representation = decode_representation[:,0:self.num_treats * self.units_per_treat]

        return latent_representation, decode_representation


    def make_decoding(self,latent_representation):

        if latent_representation is None:
            raise ValueError("Run encoder first")

        return build_inverse_fc_layer(latent_representation,self.decoder_filter_weights,self.decoder_filter_biases)

    def train(self,batch_size):


        seqs = []
        pca_seqs = []
        labels = []

        self.stds = {}
        self.means = {}
        self.lens = {}

        for index, group in data.groupby(data.tag):

            sample_num = 300
            group = group.drop(['tag'],axis=1)
            self.comp_genes = [group.columns.get_loc(c) for c in comp_gene_cols if c in group]

            group = group.as_matrix()

            #print(group.shape[1])

            self.means[index] = np.mean(group, axis=0)
            self.stds[index] = np.std(group,axis=0)
            self.lens[index] = group.shape[0]

            if sampling and int(index) in ( all_conditions - target ):
                labels.extend([int(index)] * sample_num)

                samples = []
                for k in range(sample_num):
                    take = np.random.choice(list(range(self.lens[index])),1)
                    samples.append(np.mean(group[take,:],axis=0))
                group = np.array(samples)

            else:
                labels.extend([int(index)] * self.lens[index])


            if int(index) in ( all_conditions - target ):
                pca_seqs.extend(group)

            seqs.extend(group)

        self.tms = {}
        for treat in tcs:
            m = np.zeros((1,4635))
            l = 0
            for index in tcs[treat]:
                m = m + self.means[index] * self.lens[index]
                l = l + self.lens[index]
            m = m / l
            self.tms[treat] = m

        # shuffle input


        # make input into numpy array
        seqs = np.vstack(seqs)
        pca_seqs = np.vstack(pca_seqs)

        labels = np.array(labels)

        self.pca = PCA(n_components=self.len,whiten=True)
        self.pca.fit(pca_seqs)


        seqs = self.pca.transform(seqs)

        pca_test = all_conditions
        if EVAL_PCA:
            self.pca_ratios.append(self.pca.explained_variance_ratio_)
            print(self.pca_ratios[-1])
            print(self.pca_ratios[-1].shape)
            eval_pca(self.pca,pca_test,seqs,labels,comp_gene_cols,conversion_str,comp_genes)

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels length mismatch")

        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

        for i in range(len(labels)):
            if labels[i] in all_conditions - target:
                self.train_data.append(seqs[i,:])
                index = [int(i) for i in format(labels[i],conversion_str)]
                self.train_labels.append(index)
            if labels[i] in target:
                self.test_data.append(seqs[i,:])
                index = [int(i) for i in format(labels[i],conversion_str)]
                self.test_labels.append(index)

        self.train_data, self.train_labels = shuffle_in_unison(self.train_data, self.train_labels)
        self.train_data = np.vstack(self.train_data)
        self.train_labels = np.array(self.train_labels)

        self.regs = []
        self.train_acc = 0
        self.test_acc = 0
        for i in range(num_treats):
            reg = LogisticRegressionCV()
            reg.fit(self.train_data,self.train_labels[:,i])
            self.regs.append(reg)
        for i in range(len(self.train_labels)):
            predicted = np.array([reg.predict(self.train_data[i]) for reg in self.regs]).squeeze()
            if np.sum(np.abs(predicted - self.train_labels[i])) == 0:
                self.train_acc += 1
        self.train_acc /= len(self.train_labels)

        self.train_labels = replicate_array(self.train_labels , self.units_per_treat)

        self.dec_train_labels = []
        for i in range(len(self.train_labels)):
            self.dec_train_labels.append(to_dec(self.train_labels[i], self.units_per_treat))
        self.dec_train_labels = np.array(self.dec_train_labels)

        self.test_data = np.vstack(self.test_data)
        self.test_labels = np.array(self.test_labels)

        for i in range(len(self.test_labels)):
            predicted = np.array([reg.predict(self.test_data[i]) for reg in self.regs]).squeeze()
            if np.sum(np.abs(predicted - self.test_labels[i])) == 0:
                self.test_acc += 1
        self.test_acc /= len(self.test_labels)

        self.test_labels = replicate_array(self.test_labels, self.units_per_treat)

        self.dec_test_labels = []
        for i in range(len(self.test_labels)):
            self.dec_test_labels.append(to_dec(self.test_labels[i], self.units_per_treat))
        self.dec_test_labels = np.array(self.dec_test_labels)


        num_epochs = int(sys.argv[4])

        batches_per_epoch = int(len(self.train_data) / batch_size)
        num_steps = int(num_epochs * batches_per_epoch)

        # the step at which we stop training based on the joint loss
        # and train only the prediction loss
        step_thres = int(num_epochs * batches_per_epoch) - EPOCHS_ONLY_PRED * batches_per_epoch

        init = tf.global_variables_initializer()
        self.sess.run(init)

        epoch_loss = 0

        for step in range(num_steps):
            offset = (step * batch_size) % (self.train_data.shape[0] - batch_size)
            batch_x = self.train_data[offset:(offset + batch_size), :]
            batch_y = self.train_labels[offset:(offset + batch_size),:]

            feed_dict = {self.expressions: batch_x, self.treatments: batch_y}

            if step < step_thres:
                _,joint_loss, rec_loss, pred_loss , l1_loss, l2_loss , mean, target_expr, sigma  = self.sess.run([self.train_joint,self.joint_loss,
                                                                                       self.rec_loss,self.pred_loss,self.l1_loss,
                                                                                        self.l2_loss,self.dec_out_mean, self.expressions, self.dec_out_log_sigma],
                                                                 feed_dict=feed_dict)

            else:
                _, joint_loss, rec_loss, pred_loss, = self.sess.run(
                [self.train_pred, self.joint_loss, self.rec_loss, self.pred_loss],
                feed_dict=feed_dict)

            epoch_loss += joint_loss

            if (step % batches_per_epoch == 0):
                epoch_loss /= batches_per_epoch
                epoch_loss = 0

                logging.info('joint loss: %f; rec loss %f; pred loss %f; l1 %f; l2 %f' % (joint_loss, rec_loss,pred_loss,l1_loss,l2_loss))

        train_recs, train_latent, train_decode, train_loss, train_rec_loss, train_pred_loss, train_proxy = self.sess.run([self.dec_out_mean, self.latent_representation,
                                                    self.decode_representation,self.joint_loss , self.rec_loss, self.pred_loss, self.proxy_target],
                                        feed_dict={self.expressions: self.train_data, self.treatments: self.train_labels})

        print(train_latent)

        actual_target = np.sum(self.train_labels, axis=1)
        plt.plot(train_proxy, actual_target, '*')
        plt.show()

        bs, test_recs, test_latent, test_decode, test_loss, test_rec_loss, test_pred_loss, test_proxy = self.sess.run([self.encoder_filter_biases,self.dec_out_mean,
                                            self.latent_representation, self.decode_representation, self.joint_loss , self.rec_loss, self.pred_loss, self.proxy_target],
                                        feed_dict={self.expressions: self.test_data, self.treatments: self.test_labels})

        actual_target = np.sum(self.test_labels,axis=1)
        plt.plot(test_proxy,actual_target,'*')

        if EVAL_PERF:
            test_factors = eval_perf(train_recs, train_latent, self.train_data, self.train_labels, test_recs, test_latent, self.test_data, self.test_labels,target,self.units_per_treat)

        self.generate(train_decode,self.train_labels,train_recs, test_latent)


    def generate(self,train_decode,train_labels,train_rec, test_latent, num=100):

        target_list = list(target)

        activations = {i:[] for i in list(range(self.num_treats)) + list(range(self.num_treats * self.units_per_treat, self.latent_size))}
        distrs = {i:[0,0] for i in activations}

        test_x = []
        test_y = []

        for i in range(len(train_labels)):

            for ind in range(self.num_treats):
                if train_labels[i,ind * self.units_per_treat] == 1:
                    activations[ind].append(train_decode[i,ind * self.units_per_treat : (ind + 1) * self.units_per_treat])

            for ind in range(self.num_treats * self.units_per_treat,self.latent_size):
                activations[ind].append(train_decode[i,ind])

            if train_labels[i,0] == 1 and train_labels[i,1] == 1:
                test_x.append(train_decode[i,0])
                test_y.append(train_decode[i,1])

        for i in activations:

            activations[i] = np.array(activations[i])

            #print(i)
            distrs[i][0] = np.mean(activations[i],axis = 0)
            distrs[i][1] = np.cov(activations[i],rowvar=False)


        print("Done")

        generated_data = []
        generated_labels = []

        for i in target:

            index = [int(i) for i in format(i, conversion_str)]

            arrays = []

            for j in range(self.num_treats):
                if index[j] == 1:
                    #arrays.append(np.random.multivariate_normal(distrs[j][0],distrs[j][1],(num)))
                    arrays.append(activations[j][np.random.choice(activations[j].shape[0],num,replace=True)])
                else:
                    arrays.append(np.zeros((num,self.units_per_treat)))
            for j in range(self.num_treats * self.units_per_treat,self.latent_size):
                arrays.append(np.random.normal(distrs[j][0],distrs[j][1],(100,1)))

            # study now the divergence between self.target_decode_representation and test_latent

            target_decode = np.hstack(arrays)
            should_have_decoded = test_latent[self.dec_test_labels == i]

            #compare_distributions(target_decode,should_have_decoded,self.units_per_treat,i)

            if SAMPLE_GENERATOR:
                gen_mean, gen_log_sigma = self.sess.run([self.gen_out_mean, self.gen_out_log_sigma],feed_dict={self.target_decode_representation:target_decode})

                generated = []
                for j in range(gen_mean.shape[0]):
                    generated.append(np.random.normal(gen_mean[i,:],np.exp(gen_log_sigma[i,:]),(10,gen_mean.shape[1])))

                generated = np.vstack(generated)
            else:
                generated = self.sess.run(self.gen_out_mean,
                                          feed_dict={self.target_decode_representation: target_decode})

            generated_data.extend(generated)
            generated_labels.extend([i]*num)

        generated_data = np.vstack(generated_data)
        generated_labels = 1000 + np.array(generated_labels)

        generated_indices = np.random.choice(len(generated_labels),size=len(generated_labels)//10,replace=False)
        all_generated_profiles.append(generated_data[generated_indices])
        all_generated_labels.append(np.expand_dims(generated_labels[generated_indices],1))

        training_labels = 3000 + self.dec_train_labels

        comp_genes_matrix = np.zeros((len(self.comp_genes),len(set(generated_labels)) + len(set(training_labels))))
        control_genes_matrix = np.zeros((len(self.comp_genes),len(set(generated_labels)) + len(set(training_labels))))
        target_comp_genes_matrix = np.zeros((len(self.comp_genes),len(set(generated_labels)) + len(set(training_labels))))
        self.comp_genes_xlabels = []

        df_orig = pd.DataFrame()

        flag = 0

        # Instantiate linear averaging predictions in dataframe

        for treat in self.tms:

            df_orig[treat + 2000] = self.tms[treat].squeeze()
        # Instantiate generated data

        for label in set(generated_labels.squeeze()) | set(training_labels.squeeze()) | target:

            if int(label / 1000) == 1 :
                df_orig[label] = self.pca.inverse_transform(np.mean(generated_data[generated_labels.squeeze() == label],axis=0))

                if label - 1000 in target:
                    comp_genes_matrix[:,flag] = df_orig[label][self.comp_genes]
                    control_genes_matrix[:,flag] = df_orig[label + 1000][self.comp_genes]
                    target_comp_genes_matrix[:,flag] = self.means[label-1000].squeeze()[self.comp_genes]
                    self.comp_genes_xlabels.append(str([int(i) for i in format(label-1000,conversion_str)]))
                    flag += 1
            elif int(label/1000) == 3:
                df_orig[label] = self.pca.inverse_transform(np.mean(train_rec[training_labels.squeeze() == label], axis=0))
                comp_genes_matrix[:, flag] = df_orig[label][self.comp_genes]
                control_genes_matrix[:, flag] = df_orig[label - 1000][self.comp_genes]
                target_comp_genes_matrix[:, flag] = self.means[label - 3000].squeeze()[self.comp_genes]
                self.comp_genes_xlabels.append(str([int(i) for i in format(label - 3000, conversion_str)]))
                flag += 1
            else:
                df_orig[label] = self.means[label]

        ae_profiles = []
        control_profiles = []
        target_profiles = []

        for label in target_list:

            ae_profiles.append(np.expand_dims(df_orig[label+1000],axis=1))
            control_profiles.append(np.expand_dims(df_orig[label+2000],axis=1))
            target_profiles.append(np.expand_dims(df_orig[label],axis=1))

        ae_profiles = np.hstack(ae_profiles)
        control_profiles = np.hstack(control_profiles)
        target_profiles = np.hstack(target_profiles)

        ae_profiles = scale(ae_profiles,axis = 0)
        control_profiles = scale(control_profiles,axis = 0)
        target_profiles = scale(target_profiles,axis=0)

        all_ae_profiles.append(ae_profiles)
        all_control_profiles.append(control_profiles)
        all_target_profiles.append(target_profiles)

        #print(df.corr())
        corrs = df_orig.corr()
        results = []
        for treat in target:
            results.append([treat,corrs[treat][treat+1000],corrs[treat][treat+2000]])
        results = np.array(results)
        all_results.append(results)
        print(results)


        # Analysis of Complementary Genes
        if COMP_GENES_ANALYSIS:
            comp_genes_matrix = scale(comp_genes_matrix.T).T
            control_genes_matrix = scale(control_genes_matrix.T).T
            target_comp_genes_matrix = scale(target_comp_genes_matrix.T).T

            all_comp_gene_matrices.append(comp_genes_matrix)
            all_control_matrices.append(control_genes_matrix)
            all_target_matrices.append(target_comp_genes_matrix)

        #
        X = self.pca.inverse_transform(np.vstack([generated_data, self.test_data]))
        y = np.vstack([np.expand_dims(generated_labels, 1), np.expand_dims(self.dec_test_labels, 1)])


        for j in range(5):
            a = np.random.randint(0,4635)
            print(a)

            fig, ax = plt.subplots(1, 10)
            for i in range(10):
                condition = list(target)[i]
                mask_test = (y == condition).squeeze()
                mask_gen = (y == condition + 1000).squeeze()

                ax[i].hist(X[mask_test,a],alpha=0.5,color='b',label='test')
                ax[i].hist(X[mask_gen,a],alpha=0.5,color='g',label='gen')

                print(skew(X[mask_test,a]),skew(X[mask_gen,a]))
                print(np.mean(X[mask_test,a]),np.mean(X[mask_gen,a]))

            plt.show()

        np.save('data', X)
        np.save('labels', y)

        #make_tSNE(X,y,target)

        #ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) ,geom_point(size=70, alpha=0.1) , ggtitle("tSNE dimensions colored by digit")

logging.getLogger().setLevel(logging.INFO)

train_accs = []
test_accs =[]

for modelnum in range(21):

    print(modelnum)

    units_per_treat = int(sys.argv[3])

    brian = Model(latent_size= units_per_treat * num_treats ,len=int(sys.argv[2]),num_treats=num_treats,filters=int(sys.argv[1]),units_per_treat= units_per_treat)


    brian.train(batch_size=128)

    train_accs.append(brian.train_acc)
    test_accs.append(brian.test_acc)

    if modelnum % 20 == 0 and modelnum > 0:

        #eval_aggregate_perf(brian.dec_test_labels,brian.dec_train_labels)
        if EVAL_PCA:
            pca_ratios = np.vstack(brian.pca_ratios)
            means = np.mean(pca_ratios,axis=0)
            stds = np.std(pca_ratios,axis=0)

            plt.figure()
            plt.title('Explained variance ratio per PC')
            plt.bar(list(range(pca_ratios.shape[1])), height=means, yerr=stds)
            #plt.plot(np.cumsum(means))
            print(np.cumsum(means)[-1])
            plt.show()

        np.save(str(sys.argv[5])+str(modelnum) + ',results', np.stack(all_results))

        if COMP_GENES_ANALYSIS:
            comp_genes_matrix = np.mean(np.stack(all_comp_gene_matrices),axis=0)
            control_genes_matrix = np.mean(np.stack(all_control_matrices), axis=0)
            target_comp_genes_matrix = np.mean(np.stack(all_target_matrices), axis=0)

            make_heatmap(comp_genes_matrix, control_genes_matrix, target_comp_genes_matrix, brian, comp_gene_cols)

        ae_profiles = np.mean(np.stack(all_ae_profiles),axis=0)
        control_profiles = np.mean(np.stack(all_control_profiles),axis=0)
        target_profiles = np.mean(np.stack(all_target_profiles),axis=0)


        np.save(str(sys.argv[5])+ str(modelnum) + ',profiles',np.stack([ae_profiles,control_profiles,target_profiles]))

        count = 0
        ae_ccs = []
        control_ccs = []

        for i in range(ae_profiles.shape[0]):

            ae_ccs.append(pearsonr(ae_profiles[i, :], target_profiles[i, :])[0])
            control_ccs.append(
                pearsonr(control_profiles[i, :], target_profiles[i, :])[0])
            if ae_ccs[-1] > control_ccs[-1]:
                count += 1
            # if i in brian.comp_genes:
            #     print(ae_ccs[-1],control_ccs[-1])
            #     plt.title(comp_gene_cols[brian.comp_genes.index(i)])
            #     plt.plot(ae_profiles[i,:],'b')
            #     plt.plot(control_profiles[i,:],'g')
            #     plt.plot(target_profiles[i,:],'r')
            #     plt.show()
        print(count / i)

        # generated_data = np.vstack(all_generated_profiles)
        # generated_labels = np.vstack(all_generated_labels)
        #
        #
        # X = np.vstack([generated_data, brian.test_data])
        # y = np.vstack([generated_labels, np.expand_dims(brian.dec_test_labels,1)])
        #
        #
        # make_tSNE(X,y,target)
        #
        # plt.title('Distribution of Pearson correlation coefficients')
        # plt.hist(ae_ccs,bins=100,color='b',alpha=0.4,label='model',weights=np.zeros_like(ae_ccs) + 1. / len(ae_ccs))
        # plt.hist(control_ccs,bins=100,color='g',alpha=0.4,label='control',weights=np.zeros_like(control_ccs) + 1. / len(control_ccs))
        # plt.legend()
        # plt.legend()
        # plt.show()
        # print("Gene level stats")
        # print((np.mean(ae_ccs),np.std(ae_ccs),np.mean(control_ccs),np.std(control_ccs)))

print(np.mean(train_accs),np.std(train_accs))
print(train_accs)
print(np.mean(test_accs),np.std(test_accs))
print(test_accs)