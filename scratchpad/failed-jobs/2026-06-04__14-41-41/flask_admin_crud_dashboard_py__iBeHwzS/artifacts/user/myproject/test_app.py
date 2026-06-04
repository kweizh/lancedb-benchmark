import unittest
import json
from app import app, table

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_workflow(self):
        # 1. GET / - check if seeded products are displayed
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('<table>', html)
        self.assertIn('prod-001', html)
        self.assertIn('Acme Notebook', html)
        self.assertIn('prod-005', html)
        self.assertIn('Acme Coffee Mug', html)

        # 2. GET /api/products - check if JSON endpoint works
        response = self.app.get('/api/products')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 5)
        ids = [p['id'] for p in data]
        self.assertIn('prod-001', ids)
        self.assertIn('prod-005', ids)

        # 3. GET /product/new - check form page
        response = self.app.get('/product/new')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('<form', html)
        self.assertIn('name="id"', html)

        # 4. POST /product - create a new product
        new_product = {
            'id': 'prod-006',
            'name': 'Acme Super Widget',
            'category': 'gadgets',
            'price': '49.99'
        }
        response = self.app.post('/product', data=new_product, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('prod-006', html)
        self.assertIn('Acme Super Widget', html)

        # Verify in JSON API
        response = self.app.get('/api/products')
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 6)
        prod_006 = [p for p in data if p['id'] == 'prod-006'][0]
        self.assertEqual(prod_006['name'], 'Acme Super Widget')
        self.assertEqual(prod_006['category'], 'gadgets')
        self.assertEqual(prod_006['price'], 49.99)

        # Check vector is stored deterministic
        db_rows = table.search().where("id = 'prod-006'").to_list()
        self.assertEqual(len(db_rows), 1)
        self.assertEqual(len(db_rows[0]['vector']), 32)
        v1 = db_rows[0]['vector']

        # 5. GET /product/<id>/edit - check edit form
        response = self.app.get('/product/prod-006/edit')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('Acme Super Widget', html)

        # 6. POST /product/<id> - update the product
        updated_product = {
            'name': 'Acme Mega Widget',
            'category': 'gadgets-plus',
            'price': '54.50'
        }
        response = self.app.post('/product/prod-006', data=updated_product, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn('Acme Mega Widget', html)

        # Verify in JSON API
        response = self.app.get('/api/products')
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 6)
        prod_006 = [p for p in data if p['id'] == 'prod-006'][0]
        self.assertEqual(prod_006['name'], 'Acme Mega Widget')
        self.assertEqual(prod_006['category'], 'gadgets-plus')
        self.assertEqual(prod_006['price'], 54.50)

        # 7. POST /product/<id>/delete - delete the product
        response = self.app.post('/product/prod-006/delete', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertNotIn('<td>prod-006</td>', html)

        # Verify in JSON API
        response = self.app.get('/api/products')
        data = json.loads(response.data.decode('utf-8'))
        self.assertEqual(len(data), 5)
        ids = [p['id'] for p in data]
        self.assertNotIn('prod-006', ids)

if __name__ == '__main__':
    unittest.main()
