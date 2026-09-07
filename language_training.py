import numpy as np

np.random.seed(42)

# ====================== Dataset ======================
text = """hello world. this is a tiny language model.
it learns to write simple text using attention.
hello again. the model is very small but it works.
"""
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int32)
print(f"Vocab size: {vocab_size} | Data length: {len(data)}")

# ====================== Hyperparameters ======================
seq_len = 10
d_model = 32
batch_size = 32
lr = 0.1
epochs = 800

# ====================== Parameters ======================
token_emb = np.random.randn(vocab_size, d_model) * 0.01
W = np.random.randn(d_model, vocab_size) * 0.01
b = np.zeros(vocab_size)

# ====================== Helpers ======================
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

# ====================== Forward + Backward ======================
def forward_backward(xb, yb):
    global token_emb, W, b          # <-- This fixes the error
    
    B, T = xb.shape
    
    # Embedding
    emb = token_emb[xb]                    # (B, T, d_model)
    
    # Linear layer
    logits = emb @ W + b                   # (B, T, vocab)
    
    # Softmax + Loss
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(B)[:, None], np.arange(T), yb] + 1e-9).mean()
    
    # Gradients
    dlogits = probs.copy()
    dlogits[np.arange(B)[:, None], np.arange(T), yb] -= 1
    dlogits /= (B * T)
    
    dW = np.zeros_like(W)
    for i in range(B):
        dW += emb[i].T @ dlogits[i]
    
    db = dlogits.sum(axis=(0, 1))
    demb = dlogits @ W.T
    
    # Update parameters
    for i in range(B):
        for j in range(T):
            token_emb[xb[i, j]] -= lr * demb[i, j]
    
    W -= lr * dW
    b -= lr * db
    
    return loss

# ====================== Training ======================
def get_batch():
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i+seq_len] for i in ix])
    y = np.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

print("\nTraining started...")
for epoch in range(epochs):
    xb, yb = get_batch()
    loss = forward_backward(xb, yb)
    
    if epoch % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

print("\nTraining finished!\n")

# ====================== Generation ======================
def generate(start="hello", max_new=120):
    idx = encode(start)
    for _ in range(max_new):
        context = np.array([idx[-seq_len:]])
        emb = token_emb[context]
        logits = emb @ W + b
        probs = softmax(logits[0, -1])
        next_id = np.random.choice(vocab_size, p=probs)
        idx.append(next_id)
    return decode(idx)

print("Generated samples:")
print("1 →", generate("hello"))
print("2 →", generate("the model"))
print("3 →", generate("it learns"))