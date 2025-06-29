# Training
HuggingFace's Trainer class is used for the purpose of training.
*************************************************************************************************************************

# What is Hugging Face's Trainer class?

🔧 Purpose
The Trainer class in Hugging Face 🤗 Transformers is a high-level training API designed to:
Simplify model training and evaluation
Handle boilerplate tasks like:
Training loop
Evaluation loop
Loss computation
Checkpoint saving
Gradient accumulation
Logging (e.g., with TensorBoard, Weights & Biases)
Mixed precision training
Distributed training

🛠️ When is it used?
It’s typically used for:
Fine-tuning pretrained models on downstream tasks (ASR, text classification, translation, etc.)
Rapid prototyping
*************************************************************************************************************************
# STEPS
1. Define a data collator. In contrast to most NLP models, Wav2Vec2-BERT has a much larger input length than output length. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning Wav2Vec2-BERT requires a special padding data collator, which we will define below.

2. Evaluation metric. During training, the model should be evaluated on the word error rate. We should define a compute_metrics function accordingly. "WER" is taken as an evaluation metric. Word Error Rate (WER) is a metric used to evaluate the accuracy of an automatic speech recognition (ASR) system by measuring how many words in its output differ from the correct reference transcript.

3. Load a pre-trained checkpoint. We need to load a pre-trained checkpoint and configure it correctly for training.

4. Define the training configuration.
*************************************************************************************************************************
We have customized the Wav2Vec2-BERT model for fine-tuning on OSLR 54 dataset, with:

Careful regularization (dropouts, masking)
Memory-efficient adaptation (adapters)
Proper loss handling for ASR (CTC)
Aligned tokenizer and vocabulary

⚙️ Hyperparameters Explained:
1. attention_dropout=0.1
Applies dropout to the self-attention layers in the transformer.
Prevents overfitting by randomly dropping 10% of attention weights during training.

2. hidden_dropout=0.1
Dropout applied to fully connected (dense) layers in the model.
Helps prevent overfitting.

3. feat_proj_dropout=0.1
Dropout applied to the feature projection layer that maps audio features to the transformer input dimension.
Regularizes the input to the transformer.

4. mask_time_prob=0.05
During training, 5% of time steps in the input are masked (set to 0).
This is part of the model’s data augmentation to help it learn more robust features.
Lower than default (which is often 0.065 or 0.08), meaning less aggressive masking.

5. layerdrop=0.1
Randomly drops entire transformer layers during training (like a structured dropout).
Improves generalization and robustness by forcing different combinations of layers to be active during training.

6. ctc_loss_reduction="mean"
Specifies how the Connectionist Temporal Classification (CTC) loss should be reduced.
"mean" means the loss is averaged across the batch (helps with stable training).

7. add_adapter=True
Adds adapters (small bottleneck layers) into the model to allow parameter-efficient fine-tuning.
Only the adapters are trained, not the entire model — saves memory and is faster.

8. pad_token_id=processor.tokenizer.pad_token_id
Sets the model’s padding token ID using the tokenizer from the processor.
Ensures correct alignment between audio and labels during loss computation.

9. vocab_size=len(processor.tokenizer)
Ensures the output layer (CTC head) has the correct number of output classes (i.e., number of characters/words).
Matches model output to tokenizer vocabulary size.
*************************************************************************************************************************
# Training Arguments Explained

🧠 TrainingArguments – Roles of Parameters

✅ General Setup
output_dir: Specifies where to save model checkpoints, logs, and outputs.
push_to_hub: If True, uploads the model and training artifacts to the Hugging Face Hub.

⚙️ Batching & Optimization
per_device_train_batch_size: Number of training samples per GPU (or CPU) per forward/backward pass.
gradient_accumulation_steps: Accumulates gradients over multiple steps to simulate larger batch sizes with limited memory.
gradient_checkpointing: Saves memory by recomputing activations during the backward pass instead of storing them.

🎯 Learning Rate Schedule
learning_rate: The initial learning rate used by the optimizer.
warmup_ratio: Fraction of training steps used to gradually increase the learning rate from 0 to its set value.

📊 Logging, Evaluation & Saving
eval_strategy: Defines when evaluation should happen (e.g., per step or per epoch).
eval_steps: How often (in steps) to run evaluation.
logging_steps: How often to log metrics and losses during training.
save_steps: How often to save model checkpoints.
save_total_limit: Maximum number of checkpoints to keep (old ones get deleted).

📅 Training Duration
num_train_epochs: Number of times to iterate over the full training dataset.

⚡ Training Precision
fp16: Enables mixed-precision training for faster computation and reduced memory usage.

📈 Monitoring
report_to: Specifies where to send logs and metrics (e.g., "wandb", "tensorboard").

************************************************************************************************************************
NOTE: Since, the model gave very good results in the first few experiments. Only, the number of epochs were changed for a better result.
*************************************************************************************************************************
# Minor Experiments

# Experiment 1:
-Training Arguments changed: 
           lr = 1e-5
           epoch = 7
-Result: WER = 0.34
-Notebook Link:- [Click Here](/experiment/wav2vec2-bert/wav2vec2-bert-exp-1)
-Description: Tried a new learning rate, though the training loss doesn't seem to decrease in the beginning, it didn't make a significant difference in final WER as compared to the learning rate of 3e-5

# Experiment 2:

-Whats changed: changed the dataset from this checkpoint [spktsagar/openslr-nepali-asr-cleaned](https://huggingface.co/datasets/spktsagar/openslr-nepali-asr-cleaned) to this checkpoint "[jenrish/nepali-training-data"](https://huggingface.co/datasets/jenrish/nepali-training-data) 

-Result: Validation loss infinity due to flac error. This means some of the data in jenrish is corrupted. Though the corrupted files were handled using exception handling, the model just couldn't learn. 

-Notebook Link:[Click Here](/experiment/wav2vec2-bert/wav2vec2-bert-exp-2)

-Description: This [checkpoint](https://huggingface.co/datasets/spktsagar/openslr-nepali-asr-cleaned) has the nepali numbers changed to devnagari letters(better for ASR). Now, I would recommend anyone to train this model on this jenrish data as a future enhancement of current work.

*************************************************************************************************************************
# Major Experiments

With batch_size = 4 and gradient accumulation of 2, the notebook was trained on 8, 20 and 40 epochs.
On 8 epochs, WER = 10.67 [Notebook](/experiment/wav2vec2-bert/wav2vec2-bert-exp-3)
On 20 epochs, WER = 8.7, [Notebook](/experiment/wav2vec2-bert/wav2vec2-bert-exp-4)
On 40 epochs, WER = 7.8, [Notebook](/experiment/wav2vec2-bert/wav2vec2-bert-exp-5)





