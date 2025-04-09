# Speech T5 model

SpeechT5 is a unified, sequence-to-sequence framework for speech and text processing, introduced by Microsoft Research. Inspired by the architecture of Text-to-Text Transfer Transformer (T5), it extends the T5 model to support both text and speech modalities. The model is designed to perform a wide range of tasks such as: Automatic Speech Recognition (ASR), Text-to-Speech (TTS), Speaker Identification, Speech Translation, Speech Enhancement

# Architecture
SpeechT5 leverages shared encoder-decoder architecture and introduces modality-specific adapters for handling input/output variations across tasks. Pretraining is performed on unlabeled data using self-supervised techniques, and task-specific fine-tuning enables multi-task capabilities.

📄 [Original Paper](https://arxiv.org/abs/2202.04089)

💻 [Original Model](https://github.com/microsoft/SpeechT5)

🔄 SpeechT5 Highlights

🔄 Unified Framework	 -> Handles both speech and text tasks in a single encoder-decoder model.

🧩 Modality Adapters	-> Uses task-specific adapter modules for different types of input/output (e.g., spectrograms, waveforms, text).

🔊 Multimodal Pretraining ->	Self-supervised learning on both speech and text modalities.

🧠 Multi-task Learning ->	Supports ASR, TTS, speech translation, speaker ID, and speech enhancement in a single model.

💡 Based on T5	-> Leverages the power and flexibility of the T5-style sequence-to-sequence transformer architecture.