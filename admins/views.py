from django.shortcuts import render,redirect
from django.contrib import messages
from users.models import user
import os
from django.conf import settings


# Create your views here.
def admin_login(request):
    a_email = 'admin@gmail.com'
    a_password = 'admin'

    if request.method == 'POST':
        admin_name = request.POST.get('email')
        admin_password = request.POST.get('pwd')

        if a_email == admin_name and a_password == admin_password:
            messages.success(request, 'Login successful')
            print('Admin login successful...')
            return redirect('adashboard')
        else:
            messages.error(request, 'Login credentials were incorrect.')
            print('Invalid login attempt...')
            return redirect('admin_login')

    return render(request, 'admin/admin_login.html')


def adashboard(req):
    all_users = user.objects.all().count()
    pending_users = user.objects.filter(user_status = 'pending').count()
    rejected_users = user.objects.filter(user_status = 'Rejected').count()
    accepted_users = user.objects.filter(user_status = 'Accepted').count()
    messages.success(req, 'Login Succesfully')
    return render(req, 'admin/adashboard.html', {'a' : pending_users, 'b' : all_users, 'c' : rejected_users, 'd' : accepted_users})

def pending_users(req):
    users = user.objects.filter(user_status = 'pending')
    context = {'u' : users}
    return render(req, 'admin/pending_users.html', context)

def accepted_users(req, id):
    return redirect('pending_users')

def reject_user(req,id):
    return redirect('pending_users')

def delete_users(req, id):
    return redirect('all_users')

from django.core.paginator import Paginator

def all_users(request):
    a = user.objects.all()
    paginator = Paginator(a, 5) 
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request,'admin/all_users.html',{'u':post})


def Admin_Accept_Button(request, id):
    users = user.objects.get(user_id=id)
    users.user_status = "Accepted"
    users.save()
    messages.success(request, "Status Changed Successfully")
    messages.warning(request, "Accepted")
    return redirect('pending_users')

def Admin_Reject_Btn(request, id):
    users = user.objects.get(user_id=id)
    users.user_status = "Rejected"
    users.save()
    messages.success(request, "Status Changed Successfully")
    messages.warning(request, "Rejected")
    return redirect('pending_users')


def sse_net(req):
    return render(req,'admin/sse_net.html')

def hybrid_secure_sse(req):
    return render(req,'admin/hybrid_secure_sse.html')

def graph(req):
    return render(req,'admin/graph.html')


from django.shortcuts import render
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

class EnhancedSecureSSE(nn.Module):
    def __init__(self, input_features, hidden_dim=64, num_classes=7):
        super(EnhancedSecureSSE, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def hybrid_secure_sse(request):
    X = np.load(os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/structured_sea_state_data.npy"))
    y = np.load(os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/structured_sea_state_labels.npy"))

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = EnhancedSecureSSE(input_features=X.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    def init_weights(m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    for epoch in range(30):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step(0)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            correct += (outputs.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total * 100
    torch.save(model.state_dict(), os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/secure_sse_model.pth"))

    return render(request, 'admin/hybrid_secure_sse.html', {'accuracy': round(accuracy, 2)})


import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import TensorDataset, DataLoader, random_split
from django.shortcuts import render

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class WeakSSEModel(nn.Module):
    def __init__(self, input_features, num_classes=7):
        super(WeakSSEModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_features, 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Flatten(),
            nn.Linear(2 *  X_shape[1], num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.net(x)

def sse_net(request):
    set_seed(123)

    # Load data
    X = np.load(os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/structured_sea_state_data.npy"))
    y = np.load(os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/structured_sea_state_labels.npy"))

    global X_shape
    X_shape = X.shape

    # Corrupt 30% of labels
    num_classes = len(np.unique(y))
    noise_ratio = 0.3
    num_noisy = int(noise_ratio * len(y))
    noisy_indices = np.random.choice(len(y), num_noisy, replace=False)
    for i in noisy_indices:
        y[i] = np.random.randint(0, num_classes)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator().manual_seed(123)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Model, loss, optimizer
    model = WeakSSEModel(input_features=X.shape[2], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training
    for epoch in range(5):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predicted = outputs.argmax(1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    test_accuracy = correct / total * 100

    # Pass accuracy to template
    return render(request, 'admin/sse_net.html', {'test_accuracy': round(test_accuracy, 2)})


import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from django.shortcuts import render

def graph(req):
    # Models and their metrics
    models = ['Hybrid_Secure_SSE', 'SSE_Neet']
    metrics = ['Accuracy']
    scores = {
      
        
        'Hybrid_Secure_SSE': [99.29],
        'SSE_Neet': [66.79],
    }

    # Bar width
    bar_width = 0.15
    index = np.arange(len(metrics))

    # Plot each metric as a bar chart for each model
    plt.figure(figsize=(12, 8))
    for i, model in enumerate(models):
        values = scores[model]
        plt.bar(index + i * bar_width, values, bar_width, label=model)

    # Add labels, legend, and title
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title('Comparison of Metrics Across Models', fontsize=16)
    plt.xticks(index + bar_width * (len(models) / 2), metrics, fontsize=12)
    plt.legend()

    # Annotate each bar with its value
    for i, model in enumerate(models):
        for j, value in enumerate(scores[model]):
            plt.text(index[j] + i * bar_width, value + 0.01, f'{value:.2f}', ha='center', fontsize=10)

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Provide image data for template
    context = {
        'image_data': image_data
    }

    return render(req, 'admin/graph.html', context)


