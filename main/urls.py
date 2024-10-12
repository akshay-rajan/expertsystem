from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
    path('clustering/', views.clustering, name='clustering'),
    path('linear_regression/', views.linear_regression, name='linear_regression'),
    path('knn/', views.knn, name='knn'),
    path('decision_tree/', views.decision_tree, name='decision_tree'),
    path('kmeans/', views.kmeans, name='kmeans'),
    path('samples', views.samples, name='samples'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)