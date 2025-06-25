# wav2vec2.0

Wav2Vec 2.0 is a self-supervised speech representation learning model, which is improved version of the wav2vec architecture introduced by Facebook AI (Meta).
The architecture uses a CNN-based feature encoder, a transformer encoder, and a contrastive learning objective to align masked latent representations with quantized context vectors. 

 [Original Model](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
 
 [Original Paper](https://arxiv.org/abs/2006.11477) 

 # wav2vec2 model variants with description

There are several models that are based on the wav2vec architecture.

1. Wav2Vec 2.0 - The original model with contrastive learning and Transformer encoders.

2. Wav2Vec2-CTC - Fine-tuned variant for automatic speech recognition using a CTC head.

3. XLSR - Multilingual Wav2Vec2 trained on 128 languages.

4. XLS-R - A larger, more robust version of XLSR with improved multilingual performance.

5. Wav2Vec2-BERT - Combines Wav2Vec2 as the acoustic encoder with a BERT-like decoder for sequence-to-sequence modeling.

6. MMS - Massively Multilingual Speech model built on Wav2Vec2, supports over 1,000 languages.

7. HuBERT - Learns speech units through clustering and masked prediction (successor-like evolution of Wav2Vec2).

*************************************************************************************************************************

# Wav2Vec2-BERT

Overview
The Wav2Vec2-BERT model was proposed in Seamless: Multilingual Expressive and Streaming Speech Translation by the Seamless Communication team from Meta AI [Click here to access the original paper](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/)

--Following a series of multilingual improvements (XLSR, XLS-R and MMS), Wav2Vec2-BERT is a 580M-parameters versatile audio model that has been pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. For comparison, XLS-R used almost half a million hours of audio data in 128 languages and MMS checkpoints were pre-trained on more than half a million hours of audio in over 1,400 languages. Boosting to millions of hours enables Wav2Vec2-BERT to achieve even more competitive results in speech-related tasks, whatever the language.

--Wav2Vec2-BERT follows the same architecture as Wav2Vec2-Conformer, but employs a causal depthwise convolutional layer and uses as input a mel-spectrogram representation of the audio instead of the raw waveform. Wav2Vec2-BERT also introduces a Conformer-based adapter network instead of a simple convolutional network.

--To use it for ASR, Wav2Vec2-BERT can be fine-tuned using Connectionist Temporal Classification ([CTC](https://distill.pub/2017/ctc/)), which is an algorithm that is used to train neural networks for sequence-to-sequence problems, such as ASR and handwriting recognition.

--Wav2Vec2-BERT model is thus accompanied by both a tokenizer, called Wav2Vec2CTCTokenizer, and a feature extractor, called SeamlessM4TFeatureExtractor (Critical NoteC: if you use a different tokenizer and a feature extractor,the model might run, but it won’t "learn" or infer correctly.)

--The aim of this specific documentation is to document all the elements that's needed to train Wav2Vec2-BERT model - more specifically the pre-trained checkpoint [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) - on ASR tasks, using open-source tools and models.

This amazing tutorial([Cick Here](https://huggingface.co/blog/fine-tune-w2v2-bert)) was taken as reference for fine-tuning wav2vec2-bert on OSLR 54 dataset([Click Here](https://openslr.org/54/)).

I highly recommend anyone willing to try and learn about this model and ASR in general to go through these 3 tutorials
--For data preprocessing([Click Here](https://www.spktsagar.com/posts/2022/08/finetune-xlsr-nepali/))
--For learning fine tuning pipeline for wav2vec2-BERT([Click Here](https://www.spktsagar.com/posts/2022/08/finetune-xlsr-nepali/))
--For learning about working with audio([Click Here](https://huggingface.co/learn/audio-course/en/chapter0/introduction))

