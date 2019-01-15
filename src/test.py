class Test:
    def __init__(self):
        print(hasattr(self, 'te'))
        test = getattr(self, 'te')
        test()

    def te(self):
        print('Wouaw')

bla = Test()
