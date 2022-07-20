import torch
def GETcorrectnumber(loader, printcolor, num_classes):
  with torch.no_grad():
    self.num_classes = num_claseese
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = aoemnet(inputs)
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        # n_correct += (predicted == labels).sum().item()
        for k in range(predicted.shape[0]):
          if predicted[k]==labels[k]:
            n_correct +=1
        # for i in range(num_classes): # accuracy for each class
        #     label = labels[i]
        #     pred = predicted[i]
        #     if (label == pred):
        #         n_class_correct[i] += 1
        #     n_class_samples[i] += 1
    acc = 100.0 * n_correct / n_samples
    print(printcolor+f'[{epoch + 1}] t accuracy： {acc}%'+printcolor)
  return acc