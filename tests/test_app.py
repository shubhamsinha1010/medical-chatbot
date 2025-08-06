import unittest
from app import app


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Medical Chatbot', response.data)

    def test_get(self):
        response = self.client.post('/get', data={'msg': 'What is diabetes?'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Diabetes mellitus', response.data)


if __name__ == '__main__':
    unittest.main()
