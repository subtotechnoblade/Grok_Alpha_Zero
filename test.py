# feel free to comment everything out and test your own code if you'd like
# Brian's test section
import sys
class Foo:
    @staticmethod
    def bar(x):
        return x

    def bar(self, x):
        return x

def bar(x):
    return x

if __name__ == "__main__":
    print(sys.getsizeof(Foo.bar))
    print(sys.getsizeof(bar))
