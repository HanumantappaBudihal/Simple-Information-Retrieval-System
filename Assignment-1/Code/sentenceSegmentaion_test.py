import unittest
import sentenceSegmentation as ss

class sentenceSegmentaionTestMethods(unittest.TestCase):

    def test_naive(self):
        # Arrange
        text = "Hello everyone. Welcome to IIT Madras. You are studying NLP"
        excepted_result = ['Hello everyone',
                           'Welcome to IIT Madras',
                           'You are studying NLP']
        # Act
        actual_result = ss.SentenceSegmentation().naive(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

        # text = '''(How does it deal with this parenthesis?)  "It should be part of the previous sentence." "(And the same with this one.)" ('And this one!') "('(And (this)) '?)" [(and this. )]'''
        # excepted_result = ['(How does it deal with this parenthesis?)',
        #                    '"It should be part of the previous sentence."',
        #                    '"(And the same with this one.)"',
        #                    "('And this one!')",
        #                    "\"('(And (this)) '?)\"",
        #                    '[(and this. )]' ]
        # actual_result = ss.SentenceSegmentation().naive(text)
        # # Assestion
        # self.assertEqual(excepted_result, actual_result)

    def test_punkt(self):
        # Arrange
        text = '''(How does it deal with this parenthesis?)  "It should be part of the previous sentence." "(And the same with this one.)" ('And this one!') "('(And (this)) '?)" [(and this. )]'''
        excepted_result = ['(How does it deal with this parenthesis?)',
                           '"It should be part of the previous sentence."',
                           '"(And the same with this one.)"',
                           "('And this one!')",
                           "\"('(And (this)) '?)\"",
                           '[(and this. )]' ]
        # Act
        actual_result = ss.SentenceSegmentation().punkt(text)
        # Assestion
        self.assertEqual(excepted_result, actual_result)

if __name__ == '__main__':
    unittest.main()


