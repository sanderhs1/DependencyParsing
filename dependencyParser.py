import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import ConlluDataset, collate_fn
from metric import dataset_metrics

# Disable parallel tokenization to avoid deadlocks in multi-threaded environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to map subword embeddings to word-level embeddings
def pool_subword_to_word(subword_outputs, word_to_subword_map):
    batch_size, word_seq_len = word_to_subword_map.shape
    batch_indices = torch.arange(batch_size, device=subword_outputs.device).view(batch_size, 1).expand(batch_size, word_seq_len)
    
    # Gather word representations using the mapping
    word_outputs = subword_outputs[batch_indices, word_to_subword_map, :]
    return word_outputs

# Flatten the dependencies tensors before concatenation
#all_dependencies = torch.cat([batch["dependencies"].view(-1) for batch in train_loader])

# Calculate the number of unique dependency relations
#num_dep_rels = len(torch.unique(all_dependencies))
#print(f"Number of unique dependency relations: {num_dep_rels}")


# Define the joint model for POS tagging and dependency parsing
class JointModel(nn.Module):
    def __init__(self, pretrained_model_name, num_pos_labels, num_dep_labels, max_seq_length=150):
        super(JointModel, self).__init__()

        # Load the pre-trained transformer model
        self.base_model = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
        self.hidden_size = self.base_model.config.hidden_size
        self.max_seq_length = max_seq_length

        # POS tagging classifier
        self.pos_classifier = nn.Linear(self.hidden_size, num_pos_labels)
        # Dependency relation classifier
        self.dep_relation_classifier = nn.Linear(self.hidden_size, num_dep_labels)
       
        # Dependency head projection layers
        self.dep_head_projection = nn.Linear(self.hidden_size, 128)
        self.dep_dependent_projection = nn.Linear(self.hidden_size, 128)
        
        # Root embedding for dependency parsing
        self.root_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_size))
    
    # Forward pass of the joint model
    def forward(self, input_ids, attention_mask, word_to_subword_map=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Compute logits for POS and dependency relation classification
        pos_logits = self.pos_classifier(sequence_output)
        dep_relation_logits = self.dep_relation_classifier(sequence_output)

        if word_to_subword_map is not None:
            # Aggregate subword outputs into word-level representations
            word_representations = pool_subword_to_word(sequence_output, word_to_subword_map)
            pos_logits = pool_subword_to_word(pos_logits, word_to_subword_map)
            dep_relation_logits = pool_subword_to_word(dep_relation_logits, word_to_subword_map)
            
            batch_size, seq_len, _ = word_representations.shape
            root_emb_expanded = self.root_embedding.expand(batch_size, -1, -1)
            word_representations_with_root = torch.cat([root_emb_expanded, word_representations], dim=1)
            
            # Compute dependency head scores using projected representations
            head_representations = self.dep_head_projection(word_representations_with_root)  # [batch, seq_len+1, 128]
            dependent_representations = self.dep_dependent_projection(word_representations)  # [batch, seq_len, 128]
            dep_head_scores = torch.bmm(dependent_representations, head_representations.transpose(1, 2))
            
            return pos_logits, dep_relation_logits, dep_head_scores
        else:
            return pos_logits, dep_relation_logits, None
        




# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a parser model.')
    
    # Model-related arguments
    parser.add_argument('--model', type=str, default='ltg/norbert3-xs', help='The model to use (e.g., ltg/norbert3-xs)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gradient_clipping', type=float, default=5.0, help='Gradient clipping value')
    
    # Device (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0 or cpu)')
    
    return parser.parse_args()



# Function to train the joint model for POS tagging and dependency parsing
def train_joint_model(model, train_loader, val_loader, device, epochs=10, gradient_clipping=5, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=50, 
        num_training_steps=len(train_loader) * epochs
    )
    
    for epoch in range(epochs):
        model.train()
        train_pos_loss = 0.0
        train_dep_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            pos_logits, dep_relation_logits, dep_head_scores = model(
                input_ids=batch["subword_ids"],
                attention_mask=batch["attention_mask"],
                word_to_subword_map=batch["word_to_subword_map"]
            )
            
            # Compute losses for POS tagging and dependency parsing
            pos_mask = batch["pos_tag_ids"] != -1
            dep_mask = batch["dep_relation_ids"] != -1
            
            pos_loss = F.cross_entropy(
                pos_logits.view(-1, pos_logits.size(-1))[pos_mask.view(-1)],
                batch["pos_tag_ids"].view(-1)[pos_mask.view(-1)]
            )
            
            dep_relation_loss = F.cross_entropy(
                dep_relation_logits.view(-1, dep_relation_logits.size(-1))[dep_mask.view(-1)],
                batch["dep_relation_ids"].view(-1)[dep_mask.view(-1)]
            )
            
            dep_head_loss = F.cross_entropy(
                dep_head_scores.view(-1, dep_head_scores.size(-1))[dep_mask.view(-1)],
                batch["dependencies"].view(-1)[dep_mask.view(-1)]
            )
            
            loss = pos_loss + dep_relation_loss + dep_head_loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_pos_loss += pos_loss.item()
            train_dep_loss += (dep_relation_loss + dep_head_loss).item()

            progress_bar.set_postfix(pos_loss=pos_loss.item(), dep_loss=(dep_relation_loss + dep_head_loss).item())
        
        avg_train_pos_loss = train_pos_loss / len(train_loader)
        avg_train_dep_loss = train_dep_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_pos_loss = 0.0
        val_dep_loss = 0.0
        pos_correct = 0
        dep_rel_correct = 0
        dep_head_correct = 0
        total_pos = 0
        total_dep = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                pos_logits, dep_relation_logits, dep_head_scores = model(
                    input_ids=batch["subword_ids"],
                    attention_mask=batch["attention_mask"],
                    word_to_subword_map=batch["word_to_subword_map"]
                )
                
                # Compute validation losses and metrics
                pos_mask = batch["pos_tag_ids"] != -1
                dep_mask = batch["dep_relation_ids"] != -1
                
                pos_loss = F.cross_entropy(
                    pos_logits.view(-1, pos_logits.size(-1))[pos_mask.view(-1)],
                    batch["pos_tag_ids"].view(-1)[pos_mask.view(-1)]
                )

                dep_relation_loss = F.cross_entropy(
                    dep_relation_logits.view(-1, dep_relation_logits.size(-1))[dep_mask.view(-1)],
                    batch["dep_relation_ids"].view(-1)[dep_mask.view(-1)]
                )

                dep_head_loss = F.cross_entropy(
                    dep_head_scores.view(-1, dep_head_scores.size(-1))[dep_mask.view(-1)], batch["dependencies"].view(-1)[dep_mask.view(-1)]
                )

                val_pos_loss += pos_loss.item()
                val_dep_loss += (dep_relation_loss + dep_head_loss).item()

                # Calculate accuracy for POS and dependency parsing
                _, pos_predictions = pos_logits.max(dim=-1)
                _, dep_relation_predictions = dep_relation_logits.max(dim=-1)
                _, dep_head_predictions = dep_head_scores.max(dim=-1)
                
                pos_correct += (pos_predictions[pos_mask] == batch["pos_tag_ids"][pos_mask]).sum().item()
                total_pos += pos_mask.sum().item()
                
                dep_rel_correct += (dep_relation_predictions[dep_mask] == batch["dep_relation_ids"][dep_mask]).sum().item()
                
                dep_head_correct += (dep_head_predictions[dep_mask] == batch["dependencies"][dep_mask]).sum().item()
                
                total_dep += dep_mask.sum().item()
        
        # Print validation metrics
        avg_val_pos_loss = val_pos_loss / len(val_loader)
        avg_val_dep_loss = val_dep_loss / len(val_loader)
        pos_accuracy = pos_correct / total_pos if total_pos > 0 else 0
        las = dep_rel_correct / total_dep if total_dep > 0 else 0
        uas = dep_head_correct / total_dep if total_dep > 0 else 0
        
        print(f"Epoch {epoch+1}: Avg Train POS Loss = {avg_train_pos_loss:.4f} | Avg Train DEP Loss = {avg_train_dep_loss:.4f} | Avg Val POS Loss = {avg_val_pos_loss:.4f} | Avg Val DEP Loss = {avg_val_dep_loss:.4f}")
        #print(f"Epoch {epoch+1}: POS Accuracy = {pos_accuracy:.4f} | LAS = {las:.4f} | UAS = {uas:.4f}")
    
    return model

# Function to generate predictions for given dataset using the trained model
def predict(model, input_path, output_path, tokenizer, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    dataset = ConlluDataset(
        input_path, 
        tokenizer, 
        pos_ids_to_str=state_dict['pos_vocabulary'], 
        dep_ids_to_str=state_dict['dep_vocabulary']
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    with torch.no_grad(), open(output_path, "w") as output_file:
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pos_logits, dep_relation_logits, dep_head_scores = model(
                input_ids=batch["subword_ids"],
                attention_mask=batch["attention_mask"],
                word_to_subword_map=batch["word_to_subword_map"]
            )
            
            # Get the most likely class for each position
            _, pos_predictions = pos_logits.max(dim=-1)
            _, dep_relation_predictions = dep_relation_logits.max(dim=-1)
            _, dep_head_predictions = dep_head_scores.max(dim=-1)
            
            for i in range(batch["pos_tag_ids"].size(0)):
                # Only consider valid positions (not padding)
                mask = batch["pos_tag_ids"][i] != -1
                sent_length = mask.sum().item()
                
                # Map predictions to actual POS tags using the dataset's mapping
                pos_tags = []
                for j in range(sent_length):
                    # Make sure we're getting a valid index
                    pred_idx = pos_predictions[i, j].item()
                    if 0 <= pred_idx < len(dataset.pos_ids_to_str):
                        pos_tags.append(dataset.pos_ids_to_str[pred_idx])
                    else:
                        # Fallback to a default tag if the index is invalid
                        pos_tags.append(dataset.pos_ids_to_str[0])
                
                # Process dependency predictions - format as [head, dependent] pairs
                dependencies = []
                for j in range(sent_length):
                    # Get the predicted head index
                    head_idx = dep_head_predictions[i, j].item()
                    # Ensure head index is valid (0 = root)
                    if head_idx > sent_length:
                        head_idx = 0
                    # Dependents are 1-indexed in CoNLL-U
                    dependent_idx = j + 1
                    
                    # Store only the head and dependent indices as expected by the evaluation script
                    dependencies.append([head_idx, dependent_idx])
                
                # Write the predictions to the output file
                output_file.write(json.dumps({
                    "pos_tags": pos_tags,
                    "dependencies": dependencies
                }) + "\n")
    
    print(f"Predictions saved to {output_path}")


# Main execution
if __name__ == "__main__":
    
    # Parse command-line arguments
    args = parse_arguments()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_data = ConlluDataset('/fp/projects01/ec403/IN5550/obligatories/2/no_bokmaal-ud-train.conllu', tokenizer)
    state_dict = train_data.state_dict()
    val_data = ConlluDataset('/fp/projects01/ec403/IN5550/obligatories/2/no_bokmaal-ud-dev.conllu', tokenizer)
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize and train the joint model
    joint_model = JointModel(
        pretrained_model_name="ltg/norbert3-xs",
        num_pos_labels=len(train_data.pos_ids_to_str),
        num_dep_labels=len(train_data.dep_ids_to_str)
    ).to(device)
    
    joint_model = train_joint_model(
        joint_model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        gradient_clipping=args.gradient_clipping, 
        lr=args.lr
    )

    # Predict using the joint model
    predict(
        model=joint_model,
        input_path='/fp/projects01/ec403/IN5550/obligatories/2/no_bokmaal-ud-dev.conllu',
        output_path='predictions.jsonl',
        tokenizer=tokenizer,
        device=device
    )

# Evaluate predictions
dataset_metrics("/fp/projects01/ec403/IN5550/obligatories/2/no_bokmaal-ud-dev.jsonl", "predictions.jsonl")
