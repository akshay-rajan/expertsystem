from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('introduction/', views.introduction, name='introduction'),
    path('machinelearning/', views.machinelearning, name='machinelearning'),
]
