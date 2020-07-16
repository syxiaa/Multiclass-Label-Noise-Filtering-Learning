# Multiclass-Label-Noise-Filtering-Learning

* Abstract—Label noise-containing is an important scenario in classification. As an effective method to deal with label noise, the filtering method does not need to estimate noise rate or rely on any loss function. However, most filtering-based methods are mainly designed for binary classification, and a general framework for multiple classification is lacked. To solve this problem, the definition of label noise in multiclass scenarios is given, and a general framework of label noise filtering learning method for multiclass classification is first proposed in this paper. Two instantiations of the noise filtering methods for multiclass classification, i.e., the multiclass complete random forest (mCRF) and the multiclass relative density (mRD), are derived from binary classification. Furthermore, a new cross validation for label noise filtering methods is proposed by incorporating the voting mechanism. Moreover, an adaptive and efficient method is proposed to optimize the hyper NI_threshold parameter in mCRF by using 2-means. Experiments on both the synthetic and real datasets demonstrate that the proposed noise filtering-based methods are more robust and efficient than the traditional classifiers as well as the multiclass importance reweighting method, which is the most effective method in dealing with label noise in multiclass classification as far as we known.

* mCRF.py: implements  functions of a multiple classification completely random forest  .

* mRD.py: implements the functions of multiple classification relative density.

* Adaptive.py: A fast adaptive method to find the proper NI_threshold of mCRF.

* artificial_dataset.py:  designed to do visualization.

# Requirements

### Minimal installation requirements (Python 3.7):

* Anaconda 3 
  
* Linux operating system or Windows operating system

* Sklearn ,numpy,pandas




### Installation requirements (Python 3):

* pip install -r requirements.txt


# Using

* Compare the accuracy of basic classifier, mCRF and mRD with different data sets.

# Doesn't work?

* Please contact Hao Zhou at 2294776770@qq.com
