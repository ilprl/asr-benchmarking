# wav2vec2.0 
 Wav2Vec 2.0 is a self-supervised speech representation learning model introduced by Facebook AI (Meta).

 It learns powerful audio features directly from raw waveforms without requiring transcriptions for    pretraining. The model uses a CNN-based feature encoder, a transformer encoder, and a contrastive learning objective to align masked latent representations with quantized context vectors. 
 
 It achieves state-of-the-art performance on various speech recognition benchmarks with limited labeled data.

 [Original Model](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
 
 [Original Paper](https://arxiv.org/abs/2006.11477) 

 # wav2vec2 model variants with description

1. Wav2Vec 2.0	 - The original model with contrastive learning and Transformer encoders.

2. Wav2Vec2-CTC  -	Fine-tuned variant for automatic speech recognition using a CTC head.

3. XLSR	 - Multilingual Wav2Vec2 trained on 128 languages.

4. XLS-R -	A larger, more robust version of XLSR with improved multilingual performance.

5. Wav2Vec2-BERT	- Combines Wav2Vec2 as the acoustic encoder with a BERT-like decoder for    sequence-to-sequence modeling.

6. MMS - 	Massively Multilingual Speech model built on Wav2Vec2, supports over 1,000 languages.

7. HuBERT - 	Learns speech units through clustering and masked prediction (successor-like evolution of Wav2Vec2).

8. Data2Vec	- Generalized model by Meta that unifies learning across speech, vision, and text using masked prediction.