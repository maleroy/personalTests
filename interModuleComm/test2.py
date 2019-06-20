class ClassA(object):
    def __init__(self):
        print('Hello from ClassA')
        self.my_prop_1 = 123
        self.my_prop_2 = 321

    def __str__(self):
        return "ClassA: {}-{}".format(self.my_prop_1, self.my_prop_2)
    
class ClassB(object):
    def __init__(self):
        print('Hello from ClassB')
        self.other_class = ClassA()
        self.my_prop_1 = 789
        self.my_prop_2 = 987

    def __str__(self):
        return "ClassB: {}-{} and {}".format(self.my_prop_1, self.my_prop_2, self.other_class)


