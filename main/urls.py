from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
    path('clustering/', views.clustering, name='clustering'),
]
