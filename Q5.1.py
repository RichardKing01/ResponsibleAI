import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

# Simulate some simple pre-trained word embeddings (random for this demo)
# In practice, you would use something like GloVe or Word2Vec for better embeddings.
np.random.seed(0)
def get_embedding(word):
    return torch.tensor(np.random.rand(50), dtype=torch.float32)  # Explicitly set dtype to Float

# Define the model that will combine the number and unit embeddings
class NumericUnitEmbeddings(nn.Module):
    def __init__(self, embed_size):
        super(NumericUnitEmbeddings, self).__init__()
        self.embed_size = embed_size

    def forward(self, num_tokens, unit_tokens):
        num_embeds = get_embedding(num_tokens)  # Get number embedding
        unit_embeds = get_embedding(unit_tokens)  # Get unit embedding
        return num_embeds, unit_embeds

class SimpleTransformationLayer(nn.Module):
    def __init__(self, input_size):
        super(SimpleTransformationLayer, self).__init__()
        self.fc = nn.Linear(input_size, 64)  # Transformation layer
        self.fc_out = nn.Linear(64, 1)  # Output layer (for regression task, for example)

    def forward(self, combined_embeds):
        x = torch.relu(self.fc(combined_embeds))  # Pass through transformation
        return self.fc_out(x)

# Combine the embeddings of number and unit and pass through the transformation
class NumericUnitTransformerModel(nn.Module):
    def __init__(self, embed_size):
        super(NumericUnitTransformerModel, self).__init__()
        self.embedding_layer = NumericUnitEmbeddings(embed_size)
        self.transformation_layer = SimpleTransformationLayer(embed_size * 2)  # Size doubles after concatenation

    def forward(self, num_tokens, unit_tokens):
        num_embeds, unit_embeds = self.embedding_layer(num_tokens, unit_tokens)
        combined_embeds = torch.cat([num_embeds, unit_embeds], dim=-1)  # Concatenate embeddings
        output = self.transformation_layer(combined_embeds)
        return output.squeeze()  # Ensure the output is a scalar (removes extra dimensions)

# Instantiate the model
embed_size = 50  # Dimension of each embedding (can be modified)
model = NumericUnitTransformerModel(embed_size)

# Simple test data: pairs of (numerical value, unit) and target value
# Target values are just placeholders for a regression-like task (e.g., predicting dosage)
test_data = [
    ("2", "mg", 2.0),  # e.g., "2 mg" → target value: 2.0
    ("50", "years", 50.0),  # e.g., "50 years" → target value: 50.0
    ("5", "g", 5.0),  # e.g., "5 g" → target value: 5.0
    ("100", "mg", 100.0),  # e.g., "100 mg" → target value: 100.0
]

# Training loop (simple demonstration with dummy data)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training process (for a few epochs)
for epoch in range(20):
    model.train()
    total_loss = 0
    for num_token, unit_token, target in test_data:
        optimizer.zero_grad()

        # Forward pass
        output = model(num_token, unit_token).squeeze()  # Ensure model output is a scalar
        loss = criterion(output, torch.tensor([target], dtype=torch.float32).squeeze())
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(test_data)}")

# Test the model (after training)
model.eval()
test_results = []
for num_token, unit_token, target in test_data:
    with torch.no_grad():
        output = model(num_token, unit_token)
        test_results.append((output.item(), target))

# Print out the results to see the model's performance
for output, target in test_results:
    print(f"Predicted: {output:.2f}, Actual: {target}")

# Optional: Calculate Mean Squared Error to measure the improvement
predictions = [output for output, _ in test_results]
actuals = [target for _, target in test_results]
mse = mean_squared_error(predictions, actuals)
print(f"Mean Squared Error: {mse:.4f}")