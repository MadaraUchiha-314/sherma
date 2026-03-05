from sherma import hello


def test_hello_default():
    assert hello() == "Hello, world!"


def test_hello_custom_name():
    assert hello("sherma") == "Hello, sherma!"
