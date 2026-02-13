from django.db import models

# Create your models here.
from django.db import models

class user(models.Model):
    user_id = models.AutoField(primary_key=True)
    fname = models.TextField(max_length = 20, null=True)
    lname = models.TextField(max_length=20, null=True)
    age = models.IntegerField(null=True)
    email = models.EmailField(max_length=50, null=True)
    mobile = models.TextField(max_length=20, null=True)
    password = models.TextField(max_length=20, null=True)
    user_profile =models.FileField(upload_to="images/",null=True)
    user_status = models.TextField(max_length=50, null=True, default='pending')
    user_feedback = models.TextField(max_length=1000, null=True)
    messages = models.TextField(max_length=400, null=True)

    class Meta:
        db_table = 'user_register'
