"""
URL configuration for seastate project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mains import views as m
from admins import views as a
from users import views as u
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',m.index,name='index'),
    path('about',m.about,name='about'),
    path('blog',m.blog,name='blog'),
    path('contact',m.contact,name='contact'),
    path('services',m.services,name='services'),

    #admin
    path('admin_login',a.admin_login,name='admin_login'),
    path('adashboard',a.adashboard,name='adashboard'),
    path('pending_users',a.pending_users,name='pending_users'),
    path('accepted_users',a.accepted_users,name='accepted_users'),
    path('reject_user',a.reject_user,name='reject_user'),
    path('delete_users',a.delete_users,name='delete_users'),
    path('all_users',a.all_users,name='all_users'),
    # path('Admin_Accept_Button',a.Admin_Accept_Button,name='Admin_Accept_Button'),
    path('accept_user/<int:id>/', a.Admin_Accept_Button, name='Admin_Accept_Button'),
     path('reject_user/<int:id>/', a.Admin_Reject_Btn, name='Admin_Reject_Btn'),
    # path('Admin_Reject_Btn',a.Admin_Reject_Btn,name='Admin_Reject_Btn'),
    path('sse_net',a.sse_net,name='sse_net'),
    path('hybrid_secure_sse',a.hybrid_secure_sse,name='hybrid_secure_sse'),
    path('graph',a.graph,name='graph'),


    path('user_login', u.user_login, name='user_login'),
    path('register/', u.register, name='register'),
    path('udashboard/', u.udashboard, name='udashboard'),
    path('user_profile/', u.user_profile, name='user_profile'),
    path('prediction/',u.prediction,name='prediction')
    
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


