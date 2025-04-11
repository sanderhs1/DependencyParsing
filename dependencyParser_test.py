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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=50, 
        num_training_steps=len(train_loader) * epochs
    )
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase        
        model.eval()
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
                
                # Calculate accuracy for POS and dependency parsing
                _, pos_predictions = pos_logits.max(dim=-1)
                _, dep_relation_predictions = dep_relation_logits.max(dim=-1)
                _, dep_head_predictions = dep_head_scores.max(dim=-1)
                
                pos_correct += (pos_predictions[pos_mask] == batch["pos_tag_ids"][pos_mask]).sum().item()
                total_pos += pos_mask.sum().item()
                
                dep_rel_correct += (dep_relation_predictions[dep_mask] == batch["dep_relation_ids"][dep_mask]).sum().item()
                
                dep_head_correct += (dep_head_predictions[dep_mask] == batch["dependencies"][dep_mask]).sum().item()
                
                total_dep += dep_mask.sum().item()
        
        pos_accuracy = pos_correct / total_pos if total_pos > 0 else 0
        las = dep_rel_correct / total_dep if total_dep > 0 else 0
        uas = dep_head_correct / total_dep if total_dep > 0 else 0
        
        print(f"Epoch {epoch+1}: Avg loss = {avg_train_loss:.4f}")
    
    return model

# Function to generate predictions on the test set
def predict_on_test(model, test_file_path, output_path, tokenizer, train_dataset, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Read test sentences
    test_sentences = []
    current_sentence = []
    
    with open(test_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif not line:
                if current_sentence:
                    test_sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                if parts[0].isdigit():
                    # Only store word and SpaceAfter info
                    word = parts[1].strip()
                    space_after = "SpaceAfter=No" not in line
                    current_sentence.append((word, space_after))
    
    # Add the last sentence if needed
    if current_sentence:
        test_sentences.append(current_sentence)
    
    with open(output_path, "w") as output_file:
        for sentence in tqdm(test_sentences, desc="Predicting"):
            # Prepare input for the model
            words = [word for word, _ in sentence]
            encoding = tokenizer(words, add_special_tokens=True, is_split_into_words=True, return_tensors="pt")
            
            input_ids = encoding.input_ids.to(device)
            attention_mask = encoding.attention_mask.to(device)
            
            # Create word to subword map
            word_to_subword_map = []
            for word_idx, word_id in enumerate(encoding.word_ids(batch_index=0)):
                if word_id is not None and (word_idx == 0 or encoding.word_ids(batch_index=0)[word_idx-1] != word_id):
                    word_to_subword_map.append(word_idx)
            
            word_to_subword_map = torch.tensor([word_to_subword_map]).to(device)
            
            # Make prediction
            with torch.no_grad():
                pos_logits, dep_relation_logits, dep_head_scores = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    word_to_subword_map=word_to_subword_map
                )
                
                _, pos_predictions = pos_logits.max(dim=-1)
                _, dep_relation_predictions = dep_relation_logits.max(dim=-1)
                _, dep_head_predictions = dep_head_scores.max(dim=-1)
            
            # Convert predictions to output format
            sent_length = len(sentence)
            pos_tags = [train_dataset.pos_ids_to_str[pos_predictions[0, j].item()] 
                        for j in range(sent_length)]
            
            dependencies = []
            for j in range(sent_length):
                # Get head prediction (1-indexed)
                head_idx = dep_head_predictions[0, j].item()
                if head_idx >= sent_length:
                    head_idx = 0  # Set to root if out of bounds
                dependent_idx = j + 1  # 1-indexed
                
                # Format exactly as [head_idx, dependent_idx]
                dependencies.append([head_idx, dependent_idx])
            
            # Output in the exact required format
            result = {
                "pos_tags": pos_tags,
                "dependencies": dependencies
            }
            
            output_file.write(json.dumps(result) + "\n")
    
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
    val_data = ConlluDataset('/fp/projects01/ec403/IN5550/obligatories/2/no_bokmaal-ud-dev.conllu', tokenizer)
    
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    state_dict = train_data.state_dict()
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

    # Generate predictions on the test set
    predict_on_test(
        model=joint_model,
        test_file_path='/fp/projects01/ec403/IN5550/obligatories/2/test.conllu',
        output_path='test_predictions.jsonl',
        tokenizer=tokenizer,
        train_dataset=train_data,  # Pass the training dataset to access vocabulary
        device=device
    )

