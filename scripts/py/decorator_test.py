import time


class Test(object):
    def __init__(self, func):
        print ('Init Test')
        print ('func name is {}'.format(func.__name__))
        self.__func__ = func

    def __call__(self, *args, **kwargs):
        print ("this is a wrapper")
        self.__func__()

@Test
def test():
    print('this is a test function')

test()


def deco(func):
    def wrapper():
        print("Decorating Starts.")
        temp = func()
        print("Decorating Done.")
        return temp 
    return wrapper


@deco
def foo():
    print ("This is the main func.")
    return 1


# foo()




