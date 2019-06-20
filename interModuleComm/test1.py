from test2 import ClassB

def main():
    my_test = ClassB()
    print(my_test)
    print(my_test.my_prop_1)
    print(my_test.other_class.my_prop_1)

    my_test.other_class.my_prop_2 = 654

    print(my_test)
    print(my_test.other_class)

if __name__ == '__main__':
    main()
