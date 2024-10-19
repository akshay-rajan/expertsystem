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
    path('samples', views.samples, name='samples'),
    path('preprocessing', views.preprocessing, name='preprocessing'),
    
    # path('download_csv/', views.download_csv, name='download_csv'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)