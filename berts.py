from utility.smart_batch import make_smart_batches
from utility.helper_utils import format_time, good_update_interval
from utility.bert_utils import get_tokenizer, get_model, load_model
from utility.plot_utils import plot_prcurve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import time
import os

import torch
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


class BERT_for_classification():

    def __init__(self, model_name, num_labels):
        self.tokenizer = get_tokenizer(model_name)
        self.model_name = model_name
        self.model = get_model(model_name, num_labels)

    def fit(self, batch_size=None, epochs=None, max_len=None,
            optimizer=None, test_size=None, random_state=None):
            if batch_size is None:
                batch_size = 16
            if epochs is None:
                epochs = 20
            if max_len is None:
                max_len = 512
            if optimizer is None:
                optimizer = AdamW(self.model.parameters(), lr = 5e-5, eps = 1e-8)
            
            self.batch_size = batch_size
            self.epochs = epochs
            self.max_len = max_len
            self.optimizer = optimizer

            # Total number of training steps is [number of batches] x [number of epochs]. 
            batches = len(self.X_train)//self.batch_size + 1
            total_steps = batches * self.epochs

            # Create the learning rate scheduler.
            self.scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps = 0,
                                                        num_training_steps = total_steps)

            # We'll store a number of quantities such as training and validation loss, 
            # validation accuracy, and timings.
            training_stats = []

            # Update every `update_interval` batches.
            update_interval = good_update_interval(total_iters=batches, num_desired_updates=5)

            # Measure the total training time for the whole run.
            total_t0 = time.time()

            # assign 'score' to save best model only
            best_score = 0

            # to visualize loss per each epoch
            train_loss = []
            val_loss = []

            # For each epoch...
            for epoch_i in range(1, self.epochs+1):
                self.model = self.train(update_interval, epoch_i, training_stats, train_loss)
                self.eval(self.model, epoch_i, val_loss, best_score)

            print("")
            print("Training complete!")

            print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

            report = classification_report(self.true_labels, self.preds, output_dict=True)

            if not os.path.exists('./results'):
                os.mkdir('./results')
            pd.DataFrame(report).transpose().to_csv(f'./results/{self.model_name}_epochs{self.epochs}_clf_report.csv')

            print('Classification report was saved')

            return


    def train(self, update_interval, epoch_i, training_stats, train_loss):
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, self.epochs))
        
        # At the start of each epoch (except for the first) we need to re-randomize our training data.
        # Use our `make_smart_batches` function to re-shuffle the dataset into new batches.
        (py_inputs, py_attn_masks, py_labels) = make_smart_batches(tokenizer=self.tokenizer, 
                                                                max_len=self.max_len, 
                                                                text_samples=self.X_train, 
                                                                labels=self.y_train, 
                                                                batch_size=self.batch_size)
        
        print('Training on {:,} batches...'.format(len(py_inputs)))

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        self.model.train()

        # For each batch of training data...
        for step in range(0, len(py_inputs)):

            # Progress update every, e.g., 100 batches.
            if step % update_interval == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Calculate the time remaining based on our progress.
                steps_per_sec = (time.time() - t0) / step
                remaining_sec = steps_per_sec * (len(py_inputs) - step)
                remaining = format_time(remaining_sec)

                # Report progress.
                print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}.  Remaining: {:}'.format(step, len(py_inputs), elapsed, remaining))

            # Copy the current training batch to the GPU using the `to` method.
            b_input_ids = py_inputs[step].to(self.device)
            b_input_mask = py_attn_masks[step].to(self.device)
            b_labels = py_labels[step].to(self.device)

            # Always clear any previously calculated gradients before performing a backward pass.
            self.model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The call returns the loss (because we provided labels) and the "logits"--the model outputs prior to activation.
            loss, logits = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end. 
            # `loss` is a Tensor containing a single value; 
            # the `.item()` function just returns the Python value from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0, to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"
            # how the parameters are modified based on their gradients, the learning rate, etc.
            self.optimizer.step()

            # Update the learning rate.
            self.scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(py_inputs)
        train_loss.append(avg_train_loss)
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))
            
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }
        )
        return self.model

    def eval(self, model, epoch_i, val_loss, best_score):
        # ========================================
        #               Evaluation
        # ========================================

        if epoch_i % 5 == 0: # control denominator(1~5 recommended)
            print('Predicting labels for {:,} test sentences...'.format(len(self.y_test)))

            # Put model in evaluation mode
            self.model.eval()

            # Tracking variables 
            predictions, true_labels = [], []

            # Smart Batch
            (py_inputs, py_attn_masks, py_labels) = make_smart_batches(tokenizer=self.tokenizer, 
                                                                max_len=self.max_len, 
                                                                text_samples=self.X_train, 
                                                                labels=self.y_train, 
                                                                batch_size=self.batch_size)

            # Choose an interval on which to print progress updates.
            update_interval_eval = good_update_interval(total_iters=len(py_inputs), 
                                                        num_desired_updates=10)

            # Measure elapsed time.
            t0 = time.time()
            
            # Reset the total loss for this epoch.
            total_val_loss = 0

            # For each batch of training data...
            for step in range(0, len(py_inputs)):

                # Progress update every 100 batches.
                if step % update_interval_eval == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Calculate the time remaining based on our progress.
                    steps_per_sec = (time.time() - t0) / step
                    remaining_sec = steps_per_sec * (len(py_inputs) - step)
                    remaining = format_time(remaining_sec)

                    # Report progress.
                    print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}.  Remaining: {:}'.format(step, len(py_inputs), elapsed, remaining))

                # Copy the batch to the GPU.
                b_input_ids = py_inputs[step].to(self.device)
                b_input_mask = py_attn_masks[step].to(self.device)
                b_labels = py_labels[step].to(self.device)

                # Telling the model not to compute or store gradients, saving memory and speeding up prediction
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    loss, logits = self.model(b_input_ids, 
                                            token_type_ids=None, 
                                            attention_mask=b_input_mask,
                                            labels = b_labels)
                
                total_val_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Store predictions and true labels
                predictions.append(logits)
                true_labels.append(label_ids)
                
            # Calculate the average val loss over all of the batches.
            avg_val_loss = total_val_loss / len(py_inputs)
            val_loss.append(avg_val_loss)

            # Combine the results across the batches.
            predictions = np.concatenate(predictions, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            self.true_labels = true_labels
            self.predictions = predictions

            # Choose the label with the highest score as our prediction.
            preds = np.argmax(predictions, axis=1).flatten()
            self.preds = preds

            print(classification_report(true_labels, preds))

            acc = accuracy_score(true_labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')
            print('accuracy', acc)
            print('f1(weighted)',  f1)
            print('precision', precision)
            print('recall', recall)
            print("")
            print("Average validation loss: {0:.2f}".format(avg_val_loss))
            
            if not os.path.exists('./models_BERT'):
                os.mkdir('./models_BERT')

            if f1 > best_score:
                best_score = f1

                model_dir = './models_BERT'
                state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
                }

                now = time.strftime('%m%d_%H:%M')
                torch.save(state, os.path.join(model_dir, '_'.join([self.model_name, now, str(epoch_i), str(round(f1, 4))]) + '.pth'))
                
                print('model saved')
                ## each .pth file occupies over 1GB
                ## delete .pth files in folder which are not to be used, for memory problem
        
        return

    def set_base(self, X, y, test_size=None, random_state=None):
        # Setting base variables
        if test_size is None:
            test_size=0.2
        if random_state is None:
            random_state=42

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                                    test_size=test_size, 
                                                                    random_state=random_state, 
                                                                    stratify=y)
            
        print('===========================================')
        print('Below is the shape of train/test dataset.')
        print('===========================================')
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        print('===========================================')

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            desc = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
            desc = self.model.to(self.device)
        return 


    def plot(self):
        return plot_prcurve(self.model_name, self.true_labels, self.predictions)


    def pred(self, model_file_name, model_type, batch_size=None, max_len=None):
        # Load best model
        best_model = load_model(self.model, model_file_name)

        if batch_size is None:
            batch_size = 16
        if max_len is None:
            max_len = 512

        # Create test set batch
        (py_inputs, py_attn_masks, py_labels) = make_smart_batches(tokenizer=self.tokenizer,
                                                                max_len=max_len, 
                                                                text_samples=self.X_test,  
                                                                batch_size=batch_size)
        
        print('Predicting labels for {:,} test sentences...'.format(len(self.X_test)))
        # Tracking variables 
        predictions = []

        # Choose an interval on which to print progress updates.
        update_interval = good_update_interval(total_iters=len(py_inputs), num_desired_updates=10)

        # Measure elapsed time.
        t0 = time.time()

        # Put model in prediction mode
        best_model.eval()

        # For each batch of training data...
        for step in range(0, len(py_inputs)):

            # Progress update every 100 batches.
            if step % update_interval == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Calculate the time remaining based on our progress.
                steps_per_sec = (time.time() - t0) / step
                remaining_sec = steps_per_sec * (len(py_inputs) - step)
                remaining = format_time(remaining_sec)

                # Report progress.
                print('  Batch {:>7,}  of  {:>7,}.    Elapsed: {:}.  Remaining: {:}'.format(step, len(py_inputs), elapsed, remaining))

            # Copy the batch to the GPU.
            b_input_ids = py_inputs[step].to(self.device)
            b_input_mask = py_attn_masks[step].to(self.device)
        
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = best_model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
        
            # Store predictions and true labels
            predictions.append(logits)
        print('    DONE.')

        # Combine the results across the batches.
        predictions = np.concatenate(predictions, axis=0)

        # Choose the label with the highest score as our prediction.
        preds = np.argmax(predictions, axis=1).flatten()

        df_preds = pd.DataFrame({'text': self.X_test, 'label': self.y_test, 'prediction': preds})
        
        # Record used model and date
        if not os.path.exists('./prediction'):
                os.mkdir('./prediction')
        today = time.strftime('%y%m%d')

        prediction_path = f"./prediction/{model_type}_{today}.csv"
        df_preds.to_csv(prediction_path, index=False)
        print('=====Saved prediction file=====')
        
        return predictions
