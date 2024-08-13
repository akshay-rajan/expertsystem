from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
    path('clustering/', views.clustering, name='clustering'),
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('samples', views.samples, name='samples'),
]
