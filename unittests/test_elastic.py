from mentalitystorm.elastic import ElasticQueryTool
from unittest import TestCase

class TestElasticQueryTool(TestCase):
    def test_query(self):
        eqt = ElasticQueryTool()
        eqt.queryAll()


    def test_most_improved(self):
        eqt = ElasticQueryTool()
        eqt.mostImproved()