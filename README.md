# [A novel convolutional neural network approach for classifying brain states under image stimuli](https://reveroyl.github.io/2022/08/24/thesis/)

Background: The mechanism of human neural responses to different stimuli has always been of interest to neuroscientists. In the clinical situation, tools to distinguish different diseases or states are required. However, classic classification methods have obvious shortcomings: traditional clinical categorical methods may not be competent for behaviour prediction or brain state classification and traditional machine learning models are improvable in classification accuracy. With the increasing use of convolutional neural networks (CNN) in neuroimaging computer-assisted classification, an ensemble classifier of CNNs might be able to mine hidden patterns from MEG signals. However, developing an effective brain state classifier is a difficult task owing to the non-Euclidean graphical nature of magnetoencephalography (MEG) signals.

Objective: This project had two aims: 1) to develop a CNN-based model with better performance in classification than traditional machine learning models; 2) to test if the model can be improved with extra information adding relative power spectrum.

Methods: To address this brain state classification modelling issue, I used MEG signals from 28 participants viewing 14 image stimuli to train the CNN. The CNN subsequently underwent 10-fold cross-validation to ensure proper classification of MEG. I also extracted the relative power spectrum and provided this to the network. The following main techniques were applied in this research, principal component analysis (PCA), convolutional block spatial and temporal features extracting modules, convolutional block attention module (CBAM) techniques, relative power spectrum (RPS) techniques, fully connected (FC) techniques. 

Results: In this research, my method was applied to the MEG dataset, the average classification accuracy is 23.07%±7.69%, which is much better than the baseline models: LSTM RNN model 15.38% (p = 6.8 × 10 –2) and simple image classification CNN model 11.53% (p = 5.9 × 10 –2). Relative power spectrum information (mainly beta and delta during this task) successfully informed the model improving its performance.

Conclusion: These results demonstrate that my method is feasible for the analysis and classification of brain states. It may help researchers diagnose people in the clinical situations and inform future neurological classification approaches in regard to higher specificity in identifying brain states.

![LuoLei_Poster2022_Page_2](https://raw.githubusercontent.com/ReveRoyl/PictureBed/main/BlogImg/202208241636051.png)

