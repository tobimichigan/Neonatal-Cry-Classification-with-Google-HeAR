# Neonatal-Cry-Classification-with-Google-HeAR
This research project, leveraging Google's HeAR model, classifies neonatal cries into pain, hunger, and neurological distress.


<h1><p align="center">Abstract</p></h1>
<p>Neonatal Cry Classification with Google HeAR: A Case Study Classification of Baby Cries: A Case Study Classification of Baby Cries</p>

Live Demo: https://handsonlabs.org/baby_crylensApp/crylens-app.html

<img width="1280" height="720" alt="Slide1" src="https://github.com/user-attachments/assets/09eb374a-e53b-4ca3-8798-8342e1b57acc" />



<p>Background: Neonatal crying serves as the primary communication channel for infants, encoding critical information about their physiological and emotional states. Accurate classification of cry types particularly distinguishing between pain, hunger, and neurological distress holds significant clinical potential for non-invasive assessment in neonatal intensive care units (NICU) and resource-limited settings. However, traditional cry interpretation remains subjective and inconsistent, with human accuracy reportedly as low as 34%, creating an urgent need for objective, automated screening tools.</p>

  
<p>Objective: This study presents a comprehensive machine learning pipeline for three-class neonatal cry classification (Pain, Hunger, Neurological) leveraging Google's Health Acoustic Representations (HeAR) foundation model. The HeAR encoder, pre-trained on over 300 million health-related audio clips, generates 1280-dimensional embeddings that capture rich, health-specific acoustic features optimized for medical audio analysis.
Methods: The pipeline processes 6,639 labelled cry recordings from multiple datasets, implementing a rigorous 40/15/15/30 train-validation-test-holdout split with stratification to ensure no data leakage. Audio preprocessing includes Wiener filtering, Gaussian smoothing, RMS normalization, and online augmentation (noise addition, time stretching, pitch shifting, time shifting) applied to training data only. HeAR embeddings are extracted from overlapping 2-second clips and mean-pooled per file, followed by standardization and PCA dimensionality reduction (retaining 95% variance). Four classical classifiers (SVM, Logistic Regression, Random Forest, Gradient Boosting) undergo randomized hyperparameter search, with soft-voting ensemble evaluation. A PyTorch-based deep classifier (ImprovedAttentionHead) incorporating batch normalization, dropout, Gaussian noise regularization, mixup augmentation, and cosine annealing learning rate scheduling is fine-tuned through hyperparameter search. Comprehensive evaluation includes accuracy, precision, recall, F1-score, confusion matrices, ROC curves with AUC analysis, and 5-fold cross-validation. </p>

  
<p>Results: The fine-tuned neural classifier achieved consistent performance across validation (accuracy: 0.6677, weighted F1: 0.678), test (accuracy: 0.6775, F1: 0.686), and holdout sets (accuracy: 0.6748, F1: 0.686), demonstrating robust generalization without overfitting (train-holdout gap: 0.0797). Per-class analysis revealed strong discrimination for neurological distress (AUC: 0.876, F1: 0.80), good discrimination for pain (AUC: 0.808, F1: 0.60), and fair discrimination for hunger (AUC: 0.788, F1: 0.44). The best classical model (SVM with RBF kernel) achieved comparable holdout accuracy (0.6683). Five-fold cross-validation on combined training-validation data yielded optimistic results (mean accuracy: 0.8151 ± 0.0086) due to methodological considerations.</p>

  
<p>Clinical Implications: With neurological distress AUC of 0.876, the model demonstrates good-to-very-good discriminative ability for detecting potentially serious conditions (e.g., hypoxic-ischaemic encephalopathy, seizures), enabling high-sensitivity screening protocols. Pain detection (AUC: 0.808) supports improved pain management and reduced time to analgesia. Hunger classification remains challenging (AUC: 0.788), reflecting inherent acoustic ambiguities, suggesting the need for multimodal integration (e.g., feeding schedules, vital signs). The system could serve as a clinical decision support tool for continuous NICU monitoring, bedside assessment, and retrospective analysis, potentially reducing unnecessary interventions while expediting care for high-risk infants. However, external validation on independent datasets, prospective clinical trials, regulatory approval, and explainability enhancements are required before deployment. </p>

<p>Conclusion: This HeAR-based pipeline achieves stable 67-68% accuracy in three-way neonatal cry classification, with neurological distress most reliably identified. The methodology demonstrates that pre-trained health audio representations, combined with rigorous regularization and evaluation, provide a robust foundation for developing objective, non-invasive neonatal assessment tools. The work represents a significant step toward automated cry analysis that could augment clinical judgment and improve outcomes in neonatal care. </p>

<p>Keywords
Neonatal cry classification; Health Acoustic Representations (HeAR); Google HeAR; infant cry analysis; pain detection; hunger detection; neurological distress; audio embeddings; deep learning; machine learning; transformer-based audio model; health audio foundation model; neonatal intensive care (NICU); acoustic biomarker; speech processing; audio augmentation; transfer learning; fine-tuning; support vector machine (SVM); multi-layer perceptron; ROC-AUC; class imbalance; clinical decision support; non-invasive monitoring; respiratory distress screening; baby cry analysis; paralinguistic analysis; health AI; medical audio analysis; generalizability; holdout validation </p>



h1>Gpahical Plots</h1>

<img width="1280" height="720" alt="Slide7" src="https://github.com/user-attachments/assets/8d35d583-39a1-46da-8bb5-19c1f3cb8733" />
<img width="1280" height="720" alt="Slide7" src="https://github.com/user-attachments/assets/8d35d583-39a1-46da-8bb5-19c1f3cb8733" />
<img width="1280" height="720" alt="Slide6" src="https://github.com/user-attachments/assets/28f6bb58-36e3-472c-b369-d05bf89d0935" />
<img width="1280" height="720" alt="Slide6" src="https://github.com/user-attachments/assets/28f6bb58-36e3-472c-b369-d05bf89d0935" />
<img width="1280" height="720" alt="Slide5" src="https://github.com/user-attachments/assets/b945185e-eaed-4944-a437-b816e92b27f2" />
<img width="1280" height="720" alt="Slide5" src="https://github.com/user-attachments/assets/b945185e-eaed-4944-a437-b816e92b27f2" />
<img width="1280" height="720" alt="Slide4" src="https://github.com/user-attachments/assets/c797a85d-e57d-4477-babb-a99f2e08f3e4" />
<img width="1280" height="720" alt="Slide4" src="https://github.com/user-attachments/assets/c797a85d-e57d-4477-babb-a99f2e08f3e4" />

