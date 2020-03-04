from django.contrib import admin
from django.urls import path, include
from easymoji import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('easymoji.urls')),
]
