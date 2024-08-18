import os
import time
from django.conf import settings
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Deletes files in MEDIA_ROOT that are older than 5 minutes"

    def handle(self, *args, **kwargs):
        media_root = settings.MEDIA_ROOT
        now = time.time()
        cutoff = now - 30 # 5 minutes ago

        for filename in os.listdir(media_root):
            file_path = os.path.join(media_root, filename)
            if os.path.isfile(file_path):
                if os.path.getctime(file_path) < cutoff:
                    self.stdout.write(f"Deleting {file_path}")
                    os.remove(file_path)

        self.stdout.write(f"Cleaned up {media_root}")