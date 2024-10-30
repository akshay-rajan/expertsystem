from django.contrib import admin
from .models import MLModel

class MLModelAdmin(admin.ModelAdmin):
    list_display = ('model_id', 'created_at')

admin.site.register(MLModel, MLModelAdmin)
