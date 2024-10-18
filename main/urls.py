from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
    path('clustering/', views.clustering, name='clustering'),
    path('regression/linear_regression/', views.linear_regression, name='linear_regression'),
    path('regression/lasso', views.lasso, name='lasso'),
    path('regression/ridge', views.ridge, name='ridge'),
    path('regression/decision_tree', views.decision_tree_regression, name='decision_tree_regression'),
    path('regression/random_forest', views.random_forest_regression, name='random_forest_regression'),
    path('classification/knn/', views.knn, name='knn'),
    path('classification/logistic_regression/', views.logistic_regression, name='logistic_regression'),
    path('classification/naive_bayes/', views.naive_bayes, name='naive_bayes'),
    path('classification/svm/', views.svm, name='svm'),
    path('classification/decision_tree/', views.decision_tree, name='decision_tree'),
    path('classification/random_forest/', views.random_forest, name='random_forest'),
    path('clustering/kmeans/', views.kmeans, name='kmeans'),
    path('clustering/hierarchical_clustering/', views.hierarchical_clustering, name='hierarchical_clustering'),
    path('samples', views.samples, name='samples'),
    path('predict', views.predict, name='predict'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)