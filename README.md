# WEAKLY SUPERVISED CRNN SYSTEM FOR SOUND EVENT DETECTION WITH LARGE-SCALE UNLABELED IN-DOMAIN DATA (Submitted to ICASSP2019)
# Newest update on 31/10/2018

* [DCASE2018 Task4](http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection) - This task evaluates systems for the detection of sound events in domestic environments using large-scale weakly labeled data.

* The details of our system submitted to the Task 4 of DCASE challenge 2018 are also described in the technical report titled as "A Crnn-Based System With Mixup Technique For Large-Scale Weakly Labeled Sound Event Detection". 

* Sound event detection (SED) is typically posed as a supervised learning problem requiring training data with strong temporal labels of sound events. However, the production of datasets with strong labels normally requires unaffordable labor cost. It limits the practical application of supervised SED methods. The recent advances in SED approaches focuses on detecting sound events by taking advantages of weakly labeled or unlabeled training data. In this work, we propose a joint framework to solve the SED task using large-scale unlabeled in-domain data. 

* In order to explore the possibility of making use of a large amount of unlabeled training data, we utilize our proposed system (called NUDT system) for the general audio tagging task in DCASE 2018 to predict the weak labels for unlabeled in-domain data. The code in https://github.com/Cocoxili/DCASE2018Task2/

* On the other hand, a weakly supervised architecture based on the convolutional recurrent neural network (CRNN) is developed to solve the strong annotations of sound events with the aid of the unlabeled data with predicted labels. It is found that the SED performance generally increases as more unlabeled data is added into the training. To address the noisy label problem of unlabeled data, an ensemble strategy is applied to increase the system robustness.

* In addition, a mixup technique is applied in model training process, which is believed to have some benefits on the data augmentation and the model generalization capability. Finally, the system achieves 22.05% F1-value in class-wise average metrics for the sound event detection on the provided testing dataset.

# Authors
Dezhi Wang, Kele Xu, Boqing Zhu, Lilun Zhang, Yuxing Peng,
National University of Defense Technology, Changsha, China

Thanks to Qiuqiang Kong (q.kong@surrey.ac.uk), CVSSP, University of Surrey

# Requirements
python 2.7.12

keras 2.2.0

sed-eval 0.2.1

scikit-learn 0.19

scipy 1.1.0

h5py 2.8.0

numpy 1.14

pandas 0.23

librosa 0.6.1


# Dataset

* Download the development dataset and evaluation dataset, please go to http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection.

* The annotations files and the script to download the audio files is available on the git repository for DCASE2018 task 4, go to https://github.com/DCASE-REPO/dcase2018_baseline/tree/master/task4/dataset

* In case that you experience problems during the download of the dataset please contact the task organizers by email.

# How to run

1. Modify the parameters in ./config.py if needed

2. Open ./DCASE18_Task4_SingleGPU.py and modify dataset paths to your own

3. Run ./DCASE18_Task4_SingleGPU.py to obtain sound event detection results

4. Run Convert_CSV_*.py files if csv files needed to be edited for training and evaluation

5. There is also a Multi-GPU version of the system, which can be found in DCASE18_Task4_MultiGPU.py


# Comments

* Since there is only a small weakly annotated training set is provided which is insufficient for an accurate SED in the given context, we utilize a well-developed neural network architecture in the field of computer vision to explore the possibility of making use of a large amount of unbalanced and unlabeled training data to strengthen the system performance. 

* As we have little prior knowledge of the out-of-domain unlabeled training data, this part of data is not used in our system. Also we concern that the usage of out-of-domain data could introduce noise to the training data especially the out-of-domain data does not have any reliable annotations.

* A 5-fold cross-validation is employed on the weakly labeled training dataset to train the Resnext101 model to get a convergence. In the process of weak label prediction, we set three threshold values to keep maximum 3 classes of sound events for the label of each clip in the unlabeled indomain set, which is considered reasonable for the given dataset.

* The detailed implementation of the aforementioned 'weak label predictions for unlabeled in-domain training data' can be referred to another repository developed by our team (the same for DCASE 2018 Task 2) - https://github.com/Cocoxili/DCASE2018Task2


# To be improved

* The CNN component used in CRNN framework (currently using modified ResNet50/Xception) could be improved to reduce the pooling operations in the time axis on the time-frequency inputs, which could retain more time information in the input and may produce more accurate predictions for the timestamps of sound events.

* Improve the attention/localization module to suppress the label noise of the audio segments when only clip-level labels are given in order to finally increase SED performance. 


