import argparse, os, sys, logging
import torch
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from peft import get_peft_model, PrefixTuningConfig, TaskType, PeftModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SentimentAnalysis:

    def __init__(self, modelfile, modelsuffix='.pt', basemodel='t5-base',
            epochs=10, batchsize=16, lr=5e-4, virtualtokens=30, prefixprojection=True):

        self.tokenizer = AutoTokenizer.from_pretrained(basemodel)
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.basemodel = basemodel
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = lr
        self.virtualtokens = virtualtokens
        self.prefixprojection = prefixprojection
        self.prompt = ""
        self.model = None


    def preprocess_function(self, examples):
        max_input_length = 128
        max_target_length = 4

        inputs = [f"{self.prompt}{x}" for x in examples["text"]]
        targets = [f"{x}" for x in examples["label_text"]]

        model_inputs = self.tokenizer(inputs,
                                      truncation=True,
                                      padding="max_length",
                                      max_length=max_input_length)

        labels = self.tokenizer(targets,
                                truncation=True,
                                padding="max_length",
                                max_length=max_target_length,
                                add_special_tokens=True,
                                )

        model_inputs["labels"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]]

        return model_inputs

    def get_data(self, dataset: DatasetDict):
        data_loaders = {}

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]

        train_processed = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Processing training data",
        )
        val_processed = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Processing validation data",
        )

        data_loaders["train"] = DataLoader(
            train_processed,
            batch_size=self.batchsize,
            shuffle=True,
            pin_memory=True,
            collate_fn=default_data_collator,
        )
        data_loaders["val"] = DataLoader(
            val_processed,
            batch_size=self.batchsize * 2,
            shuffle=False,
            pin_memory=True,
            collate_fn=default_data_collator,
        )

        return data_loaders

    def calculate_metrics(self, eval_preds, eval_labels):
        if len(eval_preds) != len(eval_labels):
            print(f"Warning: Prediction and label counts don't match: {len(eval_preds)} vs {len(eval_labels)}")
            min_len = min(len(eval_preds), len(eval_labels))
            eval_preds = eval_preds[:min_len]
            eval_labels = eval_labels[:min_len]

        accuracy = accuracy_score(eval_labels, eval_preds)

        metrics = {
            "accuracy": accuracy * 100,
        }

        return metrics


    def train(self, dataset: DatasetDict):
        wandb.init(project="sentiment-analysis", config={
            "base_model": self.basemodel,
            "learning_rate": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batchsize,
            "virtual_tokens": self.virtualtokens,
            "prefix_projection": self.prefixprojection
        })

        data_loaders = self.get_data(dataset)

        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.basemodel)
        base_model.resize_token_embeddings(len(self.tokenizer))

        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"Base model total parameters: {total_params:,}")

        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=self.virtualtokens,
            prefix_projection=self.prefixprojection,
            inference_mode=False,
        )

        model = get_peft_model(base_model, peft_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable parameters %: {100 * trainable_params / total_params:.4f}%")

        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(data_loaders["train"]) * self.epochs
        warmup_steps = int(0.1 * total_steps)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )


        best_val_loss = float('inf')
        best_accuracy = 0.0
        early_stopping_patience = 5
        early_stopping_counter = 0
        early_stopping_min_delta = 0.001

        for epoch in range(self.epochs):

            model.train()
            train_loss = 0.0
            loop = tqdm(data_loaders["train"], leave=True)
            steps_in_epoch = 0

            for step, batch in enumerate(loop):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                steps_in_epoch += 1


                if step % 200 == 0:
                    wandb.log({
                        "step": epoch * len(data_loaders["train"]) + step,
                        "step_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']
                    })

                    torch.cuda.empty_cache()

                loop.set_description(f"Epoch {epoch + 1}/{self.epochs}")
                loop.set_postfix(loss=loss.item())

            avg_train_loss = train_loss / steps_in_epoch
            torch.cuda.empty_cache()

            model.eval()
            val_loss = 0.0
            val_steps = 0
            eval_preds = []
            eval_labels = []

            with torch.no_grad():
                val_loop = tqdm(data_loaders["val"], leave=True, desc="Validation")
                for batch in val_loop:
                    batch_labels = batch["labels"].clone()
                    batch = {k: v.to(device) for k, v in batch.items()}

                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss
                    val_loss += loss.detach().float()
                    val_steps += 1
                    val_loop.set_postfix(loss=loss.item())


                    input_ids = batch.get("input_ids")

                    for i in range(input_ids.shape[0]):
                        input_id = input_ids[i:i + 1]
                        with torch.no_grad():
                            generated = model.generate(
                                input_ids=input_id,
                                #max_new_tokens=2,
                                max_length=2,
                                min_length=1,
                                do_sample=False,
                                num_beams=2,
                                early_stopping=True,
                                repetition_penalty=1.2,
                                eos_token_id=self.tokenizer.eos_token_id,
                            )


                        pred_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        pred_label = pred_text.strip().lower()
                        eval_preds.append(pred_label)

                    for label_ids in batch_labels.cpu().numpy():
                        label_ids = label_ids[label_ids != -100]
                        if len(label_ids) > 0:
                            decoded_label = self.tokenizer.decode(label_ids, skip_special_tokens=True)
                            eval_labels.append(decoded_label)
                        else:
                            eval_labels.append("neutral")


            if epoch % 1 == 0:
                print("\nEPOCH {} SAMPLE PREDICTIONS: ".format(epoch + 1))
                sample_size = min(10, len(eval_preds))
                for i in range(sample_size):
                    print(f"Prediction: '{eval_preds[i]}' | Ground Truth: '{eval_labels[i]}'")

                unique_preds = set(eval_preds[:100] if len(eval_preds) >= 100 else eval_preds)
                print(f"\nUnique values in the first {min(100, len(eval_preds))} predictions: {unique_preds}")

            metrics = self.calculate_metrics(eval_preds, eval_labels)


            avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
            eval_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

            print(
                f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Train PPL: {torch.exp(torch.tensor(avg_train_loss)):.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val PPL: {eval_ppl:.4f}, Accuracy: {metrics['accuracy']:.2f}%")



            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": metrics['accuracy'],
                "epoch_completed": epoch + 1,
            })

            current_accuracy = metrics['accuracy']


            if avg_val_loss < (best_val_loss - early_stopping_min_delta) or current_accuracy > best_accuracy:
                if avg_val_loss < (best_val_loss - early_stopping_min_delta):
                    best_val_loss = avg_val_loss
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy

                early_stopping_counter = 0
                best_save_path = self.modelfile + "_best" + self.modelsuffix
                model.save_pretrained(best_save_path)
                print(f"Best model saved with validation loss: {best_val_loss:.4f} and accuracy: {best_accuracy:.2f}%")
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                save_path = self.modelfile + f"_epoch{epoch + 1}" + self.modelsuffix
                model.save_pretrained(save_path)
                print(f"Model saved at {save_path}")

        print(f"Training completed. Best validation loss: {best_val_loss:.4f}, Best accuracy: {best_accuracy:.2f}%")

        wandb.log({
            "training_completed": True,
            "final_best_val_loss": best_val_loss,
            "final_best_accuracy": best_accuracy
        })

        wandb.finish()
        torch.cuda.empty_cache()

        return model

    def predict_batch(self, model, tokenizer, texts, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]

        model.eval()
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_inputs = [f"{self.prompt}{text}" for text in batch_texts]

            encoded_inputs = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **encoded_inputs,
                    max_length=2,
                    #max_new_tokens=2,
                    num_beams=2,
                    do_sample=False,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=0.6,
                    repetition_penalty=2.0,
                )

            #batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            batch_preds = [pred.strip().lower() for pred in tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            predictions.extend(batch_preds)


        return predictions

    def evaluate_model(self, model, tokenizer, test_dataset, batch_size=32):
        model.eval()

        texts = test_dataset["text"]
        labels = test_dataset["label_text"]

        predictions = self.predict_batch(model, tokenizer, texts, batch_size)

        metrics = self.calculate_metrics(predictions, labels)

        print("\n=== MODEL EVALUATION RESULTS ===")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")

        label_list = ['negative', 'neutral', 'positive']
        print(classification_report(labels, predictions, labels=label_list, digits=3, zero_division=0))

        sample_size = min(20, len(predictions))
        print("\n=== SAMPLE PREDICTIONS ===")
        for i in range(sample_size):
            print(f"Text: '{texts[i][:50]}...'")
            print(f"Prediction: '{predictions[i]}' | Ground Truth: '{labels[i]}'")
            print("-" * 40)

        return metrics



if __name__ == '__main__':

    dataset = load_dataset("ayayse/mteb-tweet-sentiment-cleaned")

    #print("Dataset structure:")
    #print(dataset)
    #print("Sample data:")
    #print(dataset["train"][0])

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-m", "--modelfile", dest="modelfile", default=os.path.join('models', 'sentiment_peft'))
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt')
    argparser.add_argument("-M", "--basemodel", dest="basemodel", default='t5-base')
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=20)
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=32)
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-4)
    argparser.add_argument("-v", "--virtualtokens", dest="virtualtokens", type=int, default=50)
    argparser.add_argument("-p", "--prefixprojection", dest="prefixprojection", action="store_true", default=True)
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False)
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None)
    argparser.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True,
                           help="Enable Weights & Biases logging")
    argparser.add_argument("--evaluate", dest="evaluate", action="store_true", default=False,
                           help="Run evaluation on test set")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if modelfile.endswith('.pt'):
        modelfile = modelfile.removesuffix('.pt')

    if not opts.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.makedirs(os.path.dirname(modelfile), exist_ok=True)

    sentiment_analysis = SentimentAnalysis(
        modelfile,
        modelsuffix=opts.modelsuffix,
        basemodel=opts.basemodel,
        epochs=opts.epochs,
        batchsize=opts.batchsize,
        lr=opts.lr,
        virtualtokens=opts.virtualtokens,
        prefixprojection=opts.prefixprojection
    )

    if not os.path.isdir(modelfile + "_best" + opts.modelsuffix) or opts.force:
        print(f"Could not find best modelfile {modelfile + '_best' + opts.modelsuffix} or -f used. Starting training.",
              file=sys.stderr)
        model = sentiment_analysis.train(dataset)
        print("Training done.", file=sys.stderr)
    else:
        print(f"Found modelfile {modelfile + '_best' + opts.modelsuffix}. Loading model.", file=sys.stderr)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(opts.basemodel)
        tokenizer = AutoTokenizer.from_pretrained(opts.basemodel)
        base_model.resize_token_embeddings(len(tokenizer))

        model_path = modelfile + "_best" + opts.modelsuffix
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)

        if opts.evaluate:
            test_dataset = dataset["test"]
            sentiment_analysis.evaluate_model(model, tokenizer, test_dataset)

        test_examples = dataset["test"][:20]
        example_texts = test_examples["text"]
        example_labels = [l for l in test_examples["label_text"]]

        predictions = sentiment_analysis.predict_batch(model, tokenizer, example_texts)

        for text, pred, true_label in zip(example_texts, predictions, example_labels):
            print(f"Tweet: {text[:128]}...")
            print(f"Prediction: {pred} | Ground Truth: {true_label}")
            print("-" * 50)