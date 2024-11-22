from django.urls import path
from . import views

urlpatterns = [
    path('introduction/', views.introduction, name='introduction'),
]
