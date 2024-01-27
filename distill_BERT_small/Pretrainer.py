import os
from transformers import AutoModelWithLMHead, AutoTokenizer, DataCollator, DataCollatorForLanguageModeling, LineByLineTextDataset, Trainer, TrainingArguments


class Pretrainer:
    def __init__(self, model_name_or_path):
        self.model = AutoModelWithLMHead.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def pretrain(self):
        os.environ["WANDB_DISABLED"] = "true"
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=.15
        )

        dataset = LineByLineTextDataset(
            tokenizer = self.tokenizer,
            file_path='./data/text.txt',
            block_size= 128 # maximum sequence length
        )

        print('No. of lines: ', len(dataset)) # No of lines in your dataset

        training_args = TrainingArguments(
            output_dir='./outputs',
            overwrite_output_dir=True,
            num_train_epochs=10,
            per_device_train_batch_size=64,
            save_steps=10000
        )

        trainer = Trainer(
            model = self.model,
            args = training_args,
            data_collator = data_collator,
            train_dataset=dataset
        )

        trainer.train()
        trainer.save_model('./outputs/')