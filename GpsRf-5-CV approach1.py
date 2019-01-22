
from __future__ import division
import numpy as np
import random
import sys,random,argparse
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
import timeit
from scipy import interp
import datetime
# similarity measures load as matrix
interact=np.loadtxt("drug_drug_matrix.txt",dtype=float)
sim1=np.loadtxt('chem_Jacarrd_sim_mat.txt',dtype=float)
sim2=np.loadtxt('enzyme_Jacarrd_sim _mat.txt',dtype=float)
sim3=np.loadtxt('indication_Jacarrd_sim_mat.txt',dtype=float)
sim4=np.loadtxt('offsideeffect_Jacarrd_sim_mat.txt',dtype=float)
sim5=np.loadtxt('pathway_Jacarrd_sim_mat.txt',dtype=float)
sim6=np.loadtxt('sideeffect_Jacarrd_sim_mat.txt',dtype=float)
sim7=np.loadtxt('target_sim_mat.txt',dtype=float)
sim8=np.loadtxt('transporter_Jacarrd_sim_mat.txt',dtype=float)

row,col=interact.shape

def mat2vec(mat):
    return list(mat.reshape((mat.shape[0]*mat.shape[1])))
link_number = 0
link_position = []
index0 = []
nonLinksPosition = []
for i in range(row):
        for j in range(col):
            if j > i:
                if interact[i][j] == 1:
                    link_position.append((i, j))
                    link_number += 1
                if interact[i][j] == 0:
                    nonLinksPosition.append((i, j))
                    index0.append(i * col + j)

labale = np.array(mat2vec(interact))

link_positionN = np.array(link_position)
nonLinksPositionN = np.array(nonLinksPosition)

def cross_validation(CV_num, seed):
    y_real = []
    y_proba = []

    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)
    act = np.zeros(interact.shape)
    fold_num = link_number // CV_num
    print(fold_num)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fo=0
    fol = 0
    for CV in range(0, 1):
        X_test=[]
        X_train=[]
        print('*********round:' + str(CV) + "**********\n")
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        test_index.sort()
        testLinkPosition = link_positionN[test_index]
        np.copyto(act,interact)
        for i in range(0, len(testLinkPosition)):
            act[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
            act[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
            testPosition = list(testLinkPosition) + list(nonLinksPosition)

        for (a,b) in testPosition:
            X_test.append(a*col+b)
                
        for i in range(row):
            for j in range(col):
                if j>i:
                    if (i*col+j) not in X_test:
                        X_train.append((i*col+j))
        X_train=X_train+index0
        
        X_test.sort()
        X_train.sort()
        
        X_test=np.array(X_test)
        X_train=np.array(X_train)
        
        X=[]
        
        k=0
        for i in range(row):
            for j in range(col):
                if j>i:
                    X.append(((i,j),k))
                    k+=1
        X=dict(X)
        
        
        x_train=[]
        x_test=[]
        for e in X_train:
            i=(e//col)
            j=e%col
            x_train.append(X[(i,j)])
        
        for e in X_test:
            i=int(e//col)
            j=e%col
            x_test.append(X[(i,j)])
            
        x_train=np.array(x_train)
        x_test=np.array(x_test)
       

# calculate score1

        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim1)+np.matmul(sim1,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim1),act))+(np.matmul(np.matmul(sim1,sim1),act))+(np.matmul(np.matmul(act,sim1),sim1))+(np.matmul(np.matmul(sim1,act),sim1))+(np.matmul(np.matmul(act,act),act))
        s1=path2+path3
        np.fill_diagonal(s1,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s1[i][j]>max1:
                    max1=s1[i][j]
        for i in range(row):
            for j in range(col):
                s1[i][j]= s1[i][j]/max1
            
# calculate score2
    
        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim2)+np.matmul(sim2,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim2),act))+(np.matmul(np.matmul(sim2,sim2),act))+(np.matmul(np.matmul(act,sim2),sim2))+(np.matmul(np.matmul(sim2,act),sim2))+(np.matmul(np.matmul(act,act),act))
        s2=path2+path3
        np.fill_diagonal(s2,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s2[i][j]>max1:
                    max1=s2[i][j]
        for i in range(row):
            for j in range(col):
                s2[i][j]= s2[i][j]/max1
            
# calculate score3

        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim3)+np.matmul(sim3,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim3),act))+(np.matmul(np.matmul(sim3,sim3),act))+(np.matmul(np.matmul(act,sim3),sim3))+(np.matmul(np.matmul(sim3,act),sim3))+(np.matmul(np.matmul(act,act),act))
        s3=path2+path3
        np.fill_diagonal(s3,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s3[i][j]>max1:
                    max1=s3[i][j]
        for i in range(row):
            for j in range(col):
                s3[i][j]= s3[i][j]/max1
            
            
# calculate score4

        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim4)+np.matmul(sim4,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim4),act))+(np.matmul(np.matmul(sim4,sim4),act))+(np.matmul(np.matmul(act,sim4),sim4))+(np.matmul(np.matmul(sim4,act),sim4))+(np.matmul(np.matmul(act,act),act))
        s4=path2+path3
        np.fill_diagonal(s4,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s4[i][j]>max1:
                    max1=s4[i][j]
        for i in range(row):
            for j in range(col):
                s4[i][j]= s4[i][j]/max1
            
# calculate score5            
    
        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim5)+np.matmul(sim5,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim5),act))+(np.matmul(np.matmul(sim5,sim5),act))+(np.matmul(np.matmul(act,sim5),sim5))+(np.matmul(np.matmul(sim5,act),sim5))+(np.matmul(np.matmul(act,act),act))
        s5=path2+path3
        np.fill_diagonal(s5,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s5[i][j]>max1:
                    max1=s5[i][j]
        for i in range(row):
            for j in range(col):
                s5[i][j]= s5[i][j]/max1
        
# calculate score6

        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim6)+np.matmul(sim6,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim6),act))+(np.matmul(np.matmul(sim6,sim6),act))+(np.matmul(np.matmul(act,sim6),sim6))+(np.matmul(np.matmul(sim6,act),sim6))+(np.matmul(np.matmul(act,act),act))
        s6=path2+path3
        np.fill_diagonal(s6,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s6[i][j]>max1:
                    max1=s6[i][j]
        for i in range(row):
            for j in range(col):
                s6[i][j]= s6[i][j]/max1
        

# calculate score7

        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim7)+np.matmul(sim7,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim7),act))+(np.matmul(np.matmul(sim7,sim7),act))+(np.matmul(np.matmul(act,sim7),sim7))+(np.matmul(np.matmul(sim7,act),sim7))+(np.matmul(np.matmul(act,act),act))
        s7=path2+path3
        np.fill_diagonal(s7,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s7[i][j]>max1:
                    max1=s7[i][j]
        for i in range(row):
            for j in range(col):
                s7[i][j]= s7[i][j]/max1
            
            
# calculate score8
        path2=np.zeros(interact.shape)
        path2=np.matmul(act,sim8)+np.matmul(sim8,act)+np.matmul(act,act)
        path3=(np.matmul(np.matmul(act,sim8),act))+(np.matmul(np.matmul(sim8,sim8),act))+(np.matmul(np.matmul(act,sim8),sim8))+(np.matmul(np.matmul(sim8,act),sim8))+(np.matmul(np.matmul(act,act),act))
        s8=path2+path3
        np.fill_diagonal(s8,0)
        max1=0
        for i in range(row):
            for j in range(col):
                if s8[i][j]>max1:
                    max1=s8[i][j]
        for i in range(row):
            for j in range(col):
                s8[i][j]= s8[i][j]/max1

#     calculate sum Score

        list_score=[]
        for i in range(row):
            for j in range(col):
                if j>i:
                    A=[round(s1[i][j],4),round(s2[i][j],4),round(s3[i][j],4),round(s4[i][j],4),round(s5[i][j],4),round(s6[i][j],4),round(s7[i][j],4),round(s8[i][j],4)]
                    
#
                    list_score.append(A)
                    A=[]
#

#  calculate

        core=np.array(list_score)
        xtrain=core[x_train]
        xtest=core[x_test]
        ytrain=labale[X_train]
        ytest=labale[X_test]

#         random.shuffle(ytrain)
    

        rf = RandomForestClassifier(n_estimators = 100)
        rf.fit(xtrain, ytrain)
        W = rf.predict_proba(xtest)[:, 1]
        # from sklearn.preprocessing import MinMaxScaler
        #
        # normalize = MinMaxScaler()
        # ensemble_prediction = normalize.fit_transform(W)
        # W=ensemble_prediction
        y_real.append(ytest)
        y_proba.append(W )

        precision, recall, pr_thresholds = precision_recall_curve(ytest, W)
        lab = 'Fold %d AUPR=%.4f' % ( fo+1, auc(recall, precision))
        fo +=1
        axes[0].step(recall, precision, label=lab)
        aupr_score = auc(recall, precision)

        all_F_measure = np.zeros(len(pr_thresholds))
        for k in range(0, len(pr_thresholds)):
            if (precision[k] + precision[k]) > 0:
                all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
            else:
                all_F_measure[k] = 0
        max_index = all_F_measure.argmax()
        threshold = pr_thresholds[max_index]

        fpr, tpr, auc_thresholds = roc_curve(ytest, W)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        axes[1].plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (fol+1, roc_auc))
        fol += 1

        predicted_score = np.zeros(len(ytest))
        predicted_score[W > threshold] = 1

        f1 = f1_score(ytest, predicted_score)
        accura = accuracy_score(ytest, predicted_score)
        pr = precision_score(ytest, predicted_score)
        rc = recall_score(ytest, predicted_score)


        print('Precision', pr)
        print('Recall', rc)
        print('F1-measure', f1)
        print('area under crov', roc_auc)
        print('area under pr-rc', aupr_score)
        print("acu", accura)

    axes[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    axes[1].plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axes[1].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    # axes[1].xlim([-0.05, 1.05])
    # axes[1].ylim([-0.05, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Roc curve ')
    axes[1].legend(loc="lower right", fontsize='small')


#     aupr curve
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    lab = 'Overall AUpr=%.4f' % (auc(recall, precision))
    axes[0].step(recall, precision, label=lab, lw=2, color='blue')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('precision-recall curve ')
    axes[0].legend(loc='lower left', fontsize='small')



    f.tight_layout()
    f.savefig('reot1.png')
    f.show()

runtimes = 1# implement 20 runs of 5-fold cross validation
a = datetime.datetime.now()
for seed in range(0,1):
    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    cross_validation(5, seed)

b = datetime.datetime.now()
c = b - a
print( int(c.total_seconds() /60))