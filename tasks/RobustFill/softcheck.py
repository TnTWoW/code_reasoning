def softcheck(actual, expected_list):
    if actual not in expected_list:
        print('softcheck failed: Got ', actual, 'but expected one of', expected_list)
