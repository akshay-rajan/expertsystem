import os
from datetime import datetime, timedelta, timezone
from django.conf import settings
from django.core.management.base import BaseCommand

from main.models import MLModel, DataFile

class Command(BaseCommand):
    help = "Deletes models saved in the database that are older than 5 minutes"

    def handle(self, *args, **kwargs):
        # Calculate the cutoff in UTC
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        # Fetch and delete the old models
        old_models = MLModel.objects.filter(created_at__lt=cutoff)
        count = old_models.count()
        old_models.delete()
        # Delete the old files
        old_files = DataFile.objects.filter(uploaded_at__lt=cutoff)
        count += old_files.count()
        old_files.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {count} files older than {cutoff}"))
