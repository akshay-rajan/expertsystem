from django.test import TestCase, Client
from django.urls import reverse
from .models import DataFile
import pandas as pd
import json
import uuid

class MainViewsTestCase(TestCase):

    def setUp(self):
        # Set up a test client
        self.client = Client()        
        # Create a sample DataFile object
        self.file_id = uuid.uuid4()
        self.data = pd.DataFrame({
            'Country': ['France', 'Spain', 'Germany'],
            'Age': [44, 27, 30],
            'Salary': [72000, 48000, 54000],
            'Purchased': ['No', 'Yes', 'No']
        })
        # Save the DataFile object to the database
        self.file_model = DataFile.objects.create(
            file_id=self.file_id,
            filename='test.csv',
            file_data=self.data.to_dict(orient='records')
        )
        # Save the file_id to the session
        self.client.session['file'] = str(self.file_id)
        self.client.session.save()

    def test_index_view(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/index.html')
        
    def test_classification_view(self):
        response = self.client.get(reverse('classification'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/algorithms.html')
        
    def test_regression_view(self):
        response = self.client.get(reverse('regression'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/algorithms.html')
        
    def test_clustering_view(self):
        response = self.client.get(reverse('clustering'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/algorithms.html')

    def test_preprocessing_view(self):
        with open('main/static/main/files/purchases.csv', 'rb') as file:
            response = self.client.post(reverse('preprocessing'), {'file': file})           
        self.assertEqual(response.status_code, 200)
        self.assertIn('json_data', response.json())
