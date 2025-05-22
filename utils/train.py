def train_model(model, loss_fn, optimizer, num_epochs,train_loader):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_Y in train_loader:
          output = model(batch_X)
          loss = loss_fn(output, batch_Y)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")