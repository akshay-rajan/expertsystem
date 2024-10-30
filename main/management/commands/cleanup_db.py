import os
from datetime import datetime, timedelta, timezone
from django.conf import settings
from django.core.management.base import BaseCommand

from main.models import MLModel

class Command(BaseCommand):
    help = "Deletes models saved in the database that are older than 5 minutes"

    def handle(self, *args, **kwargs):
        # Calculate the cutoff in UTC
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        # Fetch and delete the old models
        old_models = MLModel.objects.filter(created_at__lt=cutoff)
        old_models.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted models older than {cutoff}"))
