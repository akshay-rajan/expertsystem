import uuid
import pickle
from django.db import models

# Stores all the trained models
class MLModel(models.Model):
    # Unique identifier for the model, which is also stored in the user's session
    model_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, unique=True)
    # Serialized model
    model_data = models.BinaryField()
    
    def __str__(self):
        return str(self.model_id)
    
    def save_model(self, model):
        """Serialize and save the model to the database"""
        self.model_data = pickle.dumps(model)
        self.save()
        
    def load_model(self):
        """Deserialize the model from the database and return it"""
        return pickle.loads(self.model_data)

