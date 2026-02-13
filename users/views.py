from django.shortcuts import render
from django.shortcuts import render,redirect
from users.models import user
from django.http import JsonResponse
import os
from django.conf import settings


def register(req):

    if req.method == 'POST':
        user_fname = req.POST.get('fname')
        user_lname = req.POST.get('lname')
        user_age = req.POST.get('age')
        user_mobile = req.POST.get('mobile')
        user_password = req.POST.get('pwd')
        user_email = req.POST.get('email')
        users_image = req.FILES.get('profile')
        print(user_fname, user_lname, user_age, user_mobile, user_password, users_image,user_email)

        try:
            user.objects.get(email = user_email)
            return redirect('register')
        except user.DoesNotExist:
            user.objects.create(
                fname = user_fname,
                lname = user_lname,
                password = user_password,
                age = user_age,
                email = user_email,
                mobile = user_mobile,
                user_profile = users_image
            )
            req.session ['email'] = user_email
            return redirect('user_login')

    return render(req, 'users/register.html')

def user_login(req):
    if req.method == 'POST':
        user_email = req.POST.get('email') 
        user_password = req.POST.get('pwd')
        print(user_email, user_password)

        try:
            user_details = user.objects.get(email=user_email)
            if user_details.password == user_password:
                if user_details.user_status == 'Accepted':
                    req.session['user_id'] = user_details.user_id
                    return redirect('udashboard')
                else:
                    error = f"Your account status is '{user_details.user_status}'. Please wait for admin approval."
                    return render(req, 'users/user_login.html', {'error': error})
            else:
                return render(req, 'users/user_login.html', {'error': 'Invalid password. Please try again.'})
        except user.DoesNotExist:
            return render(req, 'users/user_login.html', {'error': 'Email not registered. Please create an account first.'})
        
    return render(req, 'users/user_login.html')


def udashboard(req):
        return render(req, 'users/udashboard.html')

from django.contrib import messages


def user_profile(request):
    views_id = request.session['user_id']
    users = user.objects.get(user_id = views_id)
    if request.method =='POST':
        userfname = request.POST.get('f_name')
        userlname = request.POST.get('l_name')
        email = request.POST.get('email_address')
        phone = request.POST.get('Phone_number')
        password = request.POST.get('pass')
        age = request.POST.get('age')
        print(userfname, userlname, email, phone, password, age)

        users.fname = userfname
        users.lname = userlname
        users.email = email
        users.mobile = phone
        users.password = password
        users.age = age

        if len(request.FILES)!= 0:
            image = request.FILES['image']
            users.user_profile = image
            users.fname = userfname
            users.lname = userlname
            users.email = email
            users.mobile = phone
            users.password = password
            users.age = age

            users.save()
            messages.success(request, 'Updated Successfully...!')

        else:
            users.fname = userfname
            users.lname = userlname
            users.email = email
            users.mobile = phone
            users.password = password
            users.age = age
            users.save()
            messages.success(request, 'Updated Successfully...!')

    return render(request,'users/user_profile.html', {'i':users})


# Create your views here.
def  prediction(req):
    return render(req,'users/prediction.html')






# import torch
# import torch.nn as nn
# import numpy as np
# from django.shortcuts import render

# # Model definition
# class EnhancedSecureSSE(nn.Module):
#     def __init__(self, input_features, hidden_dim=64, num_classes=7):
#         super(EnhancedSecureSSE, self).__init__()
#         self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
#         self.pool = nn.AdaptiveMaxPool1d(1)
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x).squeeze(-1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

# # Class mapping
# mapped_classes = {
#     0: 1,  # Original class 0 -> Custom class 1
#     3: 2,  # Original class 3 -> Custom class 2
#     6: 3   # Original class 6 -> Custom class 3
# }

# # Description mapping
# custom_descriptions = {
#     1: ("Calm (Rippled)", "Ripples with no foam crests."),
#     2: ("Slight", "Small waves, breaking with glassy crests."),
#     3: ("Very Rough", "Large waves, extensive foam.")
# }

# # Load the model once
# model = EnhancedSecureSSE(input_features=10)
# model.load_state_dict(torch.load("E:/seastate/Sea_dataset/Sea_dataset/secure_sse_model.pth", map_location=torch.device("cpu")))
# model.eval()

# def prediction(request):
#     if request.method == "POST":
#         try:
#             # Read descriptive inputs from form
#             user_input = [
#                 float(request.POST.get("wave_height")),
#                 float(request.POST.get("swell_period")),
#                 float(request.POST.get("wind_speed")),
#                 float(request.POST.get("temperature")),
#                 float(request.POST.get("salinity")),
#                 float(request.POST.get("current_speed")),
#                 float(request.POST.get("pressure")),
#                 float(request.POST.get("turbidity")),
#                 float(request.POST.get("chlorophyll")),
#                 float(request.POST.get("sea_surface_elevation")),
#             ]

#             # Convert to tensor
#             input_array = np.array(user_input, dtype=np.float32).reshape(1, 1, -1)
#             input_array = np.repeat(input_array, 2000, axis=1)
#             input_tensor = torch.tensor(input_array)

#             # Predict
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 original_class = torch.argmax(output, dim=1).item()

#             # Map prediction
#             if original_class in mapped_classes:
#                 custom_class = mapped_classes[original_class]
#                 label, description = custom_descriptions[custom_class]
#             else:
#                 label = "Unknown"
#                 description = "Predicted class is not among the mapped classes."

#             return render(request, "users/prediction.html", {
#                 "label": label,
#                 "description": description
#             })

#         except Exception as e:
#             return render(request, "users/prediction.html", {
#                 "label": "Error",
#                 "description": f"Something went wrong: {str(e)}"
#             })

#     return render(request, "users/prediction.html")




import torch
import torch.nn as nn
import numpy as np
from django.shortcuts import render

# Define model architecture
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

# Mapping original model classes to custom labels
mapped_classes = {
    0: 1,  # Original class 0 -> Custom class 1
    3: 2,  # Original class 3 -> Custom class 2
    6: 3   # Original class 6 -> Custom class 3
}

# Descriptions for mapped custom class labels
custom_descriptions = {
    1: ("Calm (Rippled)", "Ripples with no foam crests."),
    2: ("Slight", "Small waves, breaking with glassy crests."),
    3: ("Very Rough", "Large waves, extensive foam.")
}

# Prediction view
def prediction(request):
    label, description = None, None

    if request.method == "POST":
        try:
            # Get user inputs from form
            user_input = [
                float(request.POST.get("wave_height")),
                float(request.POST.get("swell_period")),
                float(request.POST.get("wind_speed")),
                float(request.POST.get("temperature")),
                float(request.POST.get("salinity")),
                float(request.POST.get("current_speed")),
                float(request.POST.get("pressure")),
                float(request.POST.get("turbidity")),
                float(request.POST.get("chlorophyll")),
                float(request.POST.get("sea_elevation"))
            ]

            # Prepare input tensor
            input_array = np.array(user_input, dtype=np.float32).reshape(1, 1, -1)
            input_array = np.repeat(input_array, 2000, axis=1)
            input_tensor = torch.tensor(input_array)

            # Load model
            model = EnhancedSecureSSE(input_features=10)
            model.load_state_dict(torch.load(os.path.join(settings.BASE_DIR, "Sea_dataset/Sea_dataset/secure_sse_model.pth"), map_location=torch.device('cpu')))
            model.eval()

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                original_class = torch.argmax(output, dim=1).item()

            # Map prediction
            if original_class in mapped_classes:
                custom_class = mapped_classes[original_class]
                label, description = custom_descriptions[custom_class]
            else:
                label = "Unmapped Class"
                description = "Predicted class is not among the mapped classes (0, 3, 6)."

        except Exception as e:
            label = "Error"
            description = f"An error occurred: {str(e)}"

    return render(request, "users/prediction.html", {
        "label": label,
        "description": description
    })

