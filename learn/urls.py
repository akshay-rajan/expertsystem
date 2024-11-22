from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='learn'),
    path('<slug:chapter>/', views.chapter_view, name='chapter'),
]
