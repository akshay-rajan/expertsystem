import uuid
import pickle
from django.db import models

# Stores all the trained models
class MLModel(models.Model):
    # Unique identifier for the model, which is also stored in the user's session
    model_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    # Serialized model
    model_data = models.BinaryField()
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return str(self.model_id)
    
    def save_model(self, model):
        """Serialize and save the model to the database"""
        self.model_data = pickle.dumps(model)
        self.save()
        
    def load_model(self):
        """Deserialize the model from the database and return it"""
        return pickle.loads(self.model_data)

# Store all uploaded and processed files
class DataFile(models.Model):
    # Unique identifier for the file
    file_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    # File name
    filename = models.CharField(max_length=255)
    # File content
    file_data = models.JSONField()
    # Timestamp
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.file_name
    
    def save_file(self, filename, file):
        """Save the file to the database as JSON"""
        self.filename = filename
        self.file_data = file
        self.save()
        
    def load_file(self):
        """Return the file from the database"""
        return self.filename, self.file_data
